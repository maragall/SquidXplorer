"""The two operator panels in PANE 1: their POLICY, separately from their pixels.

Covers the decisions those panels make — which kwargs a registration/fusion run is launched
with, what the scope selector offers, and when an operator must refuse with a sentence
instead of running. Pure functions over plain data throughout; the Qt half (that the widgets
build and are wired to these functions) is at the bottom and runs offscreen.
"""

from __future__ import annotations

import numpy as np
import pytest

from squidxplorer._op_panels import (
    STITCH_DEFAULTS,
    stitch_refusal,
    stitch_operator_kwargs,
)


# ---------------------------------------------------------------------------------------
# scope: ONE control surface, and it is NOT on the operator panel
# ---------------------------------------------------------------------------------------
# Scope belongs to the RUN, not the operator: `_run_scope.resolve_run_scope` is the single
# owner and pane 1's "run on" selector is its control (see tests/test_run_scope.py and
# test_the_panel_does_not_carry_its_own_scope below).

# ---------------------------------------------------------------------------------------
# the stitcher's control surface -> stitch_region's kwargs
# ---------------------------------------------------------------------------------------

def test_defaults_reproduce_the_pipeline_exactly():
    """An untouched panel must launch byte-for-byte what stitch_region does unaided."""
    from squidxplorer._stitch import _ABS_THRESH, _BLEND_PX, _REL_THRESH

    kw = stitch_operator_kwargs(**STITCH_DEFAULTS)
    assert kw["blend_px"] == _BLEND_PX
    assert kw["rel_thresh"] == pytest.approx(_REL_THRESH)
    assert kw["abs_thresh"] == pytest.approx(_ABS_THRESH)
    assert kw["register"] is True
    assert kw["channels"] is None                 # all channels
    assert kw["registration_channel"] == 0        # the first — None spelled concretely


def test_the_outlier_percentage_becomes_a_fraction():
    """The UI shows a percentage; ``two_round_optimization`` wants a fraction, converted here once."""
    kw = stitch_operator_kwargs(**{**STITCH_DEFAULTS, "outlier_rel_pct": 25})
    assert kw["rel_thresh"] == pytest.approx(0.25)


def test_registration_off_drops_the_registration_only_knobs():
    """With register=False there is no pose graph, so a blunder threshold is meaningless."""
    kw = stitch_operator_kwargs(**{**STITCH_DEFAULTS, "register": False})
    assert kw["register"] is False
    assert "rel_thresh" not in kw and "abs_thresh" not in kw
    assert "registration_channel" not in kw
    assert kw["blend_px"] == STITCH_DEFAULTS["blend_px"]   # fusion still feathers


def test_a_channel_subset_is_passed_through_as_indices():
    kw = stitch_operator_kwargs(**{**STITCH_DEFAULTS, "channels": [0, 2]})
    assert kw["channels"] == [0, 2]


def test_selecting_every_channel_is_spelled_None_not_a_full_list():
    """``stitch_region`` documents None as "all"; an explicit full list would be a second
    spelling of the same intent."""
    kw = stitch_operator_kwargs(**{**STITCH_DEFAULTS, "channels": [0, 1, 2]}, n_channels=3)
    assert kw["channels"] is None


def test_an_empty_channel_selection_is_refused_rather_than_fusing_nothing():
    with pytest.raises(ValueError, match="at least one channel"):
        stitch_operator_kwargs(**{**STITCH_DEFAULTS, "channels": []})


def test_a_blend_wider_than_the_tile_is_refused():
    """A feather ramp wider than the overlap never reaches full weight and dims the seam."""
    with pytest.raises(ValueError, match="blend"):
        stitch_operator_kwargs(**{**STITCH_DEFAULTS, "blend_px": 4096}, tile_px=2084)


def test_the_kwargs_are_accepted_by_stitch_region_itself():
    """Every key the panel emits must be a real parameter of ``stitch_region``."""
    import inspect

    from squidxplorer._stitch import stitch_region

    accepted = set(inspect.signature(stitch_region).parameters)
    for case in ({}, {"register": False}, {"channels": [0]}):
        kw = stitch_operator_kwargs(**{**STITCH_DEFAULTS, **case})
        assert set(kw) <= accepted, f"not parameters of stitch_region: {set(kw) - accepted}"


# ---------------------------------------------------------------------------------------
# the stitch guard, surfaced BEFORE the run
# ---------------------------------------------------------------------------------------

def test_a_labels_operator_is_refused_with_a_sentence_naming_the_way_out(blob_operator):
    """``stitch_region`` raises for a labels operator; the panel asks the same registry first
    rather than let the user discover it after a multi-minute run."""
    why = stitch_refusal(blob_operator)
    assert why is not None
    assert blob_operator in why
    assert "label" in why.lower()
    assert "per FOV" in why or "intensity" in why    # it must say what to do instead


def test_an_intensity_operator_is_not_refused(identity_operator):
    """Per-plane fusion made a plane-op stitchable; a pre-check outliving that change would be
    the engine's answer, wrong, delivered with authority."""
    assert stitch_refusal(identity_operator) is None
    assert stitch_refusal("decon") is None


def test_a_z_reducer_is_not_refused():
    assert stitch_refusal("mip") is None


def test_an_unknown_operator_is_named_rather_than_crashing_the_panel():
    why = stitch_refusal("does_not_exist")
    assert why is not None and "does_not_exist" in why


# ---------------------------------------------------------------------------------------
# the deconvolution QC verdict (the "add one more iteration?" decision)
# ---------------------------------------------------------------------------------------

def test_the_first_iteration_has_nothing_to_compare_against():
    from squidxplorer._decon_qc import halo_verdict

    kind, msg = halo_verdict([(2, 0.40)])
    assert kind == "first"
    assert "0.40" in msg or "0.4" in msg


def test_a_falling_ratio_says_the_halo_is_still_tightening():
    from squidxplorer._decon_qc import halo_verdict

    kind, msg = halo_verdict([(2, 0.40), (3, 0.31)])
    assert kind == "improving"
    assert "another" in msg.lower() or "more" in msg.lower()


def test_a_rising_ratio_says_the_disc_is_growing_back():
    """The semi-convergence tell must be stated as "stop / go back", not a neutral number."""
    from squidxplorer._decon_qc import halo_verdict

    kind, msg = halo_verdict([(2, 0.31), (3, 0.44)])
    assert kind == "worse"
    assert "2" in msg                       # names the iteration to go back to


def test_the_verdict_uses_the_best_seen_not_merely_the_previous_one():
    """Falling, falling, rising, rising: the answer is still "the best was k=3"."""
    from squidxplorer._decon_qc import halo_verdict

    kind, msg = halo_verdict([(1, 0.9), (2, 0.5), (3, 0.3), (4, 0.4), (5, 0.6)])
    assert kind == "worse"
    assert "3" in msg


def test_an_empty_history_is_refused_rather_than_returning_a_confident_nothing():
    from squidxplorer._decon_qc import halo_verdict

    with pytest.raises(ValueError):
        halo_verdict([])


# ---------------------------------------------------------------------------------------
# Qt: the widgets build, and their buttons are wired to the policy above
# ---------------------------------------------------------------------------------------

import os  # noqa: E402

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

import sys  # noqa: E402

if "PySide6" in sys.modules or "PySide2" in sys.modules:   # pragma: no cover
    pytest.skip("a PySide binding is already loaded", allow_module_level=True)

from qtpy.QtCore import Qt  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from squidxplorer._op_panels import DeconQCPanel, DeconQCResultView, StitcherPanel  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class _Host:
    """The slice of PlateWindow the panels actually use, kept small on purpose."""

    def __init__(self, channels=("c0", "c1"), order=("A1", "A2")):
        self.calls = []
        self.said = []
        self.published = []
        self._order = list(order)
        self._selected_regions = []
        self._meta = {"channels": [{"name": c} for c in channels],
                      "frame_shape": (256, 256), "regions": list(order),
                      "fovs_per_region": {r: [0, 1] for r in order},
                      "pixel_size_um": 1.0, "dz_um": 1.5, "dtype": "uint16",
                      "z_levels": [0, 1], "n_t": 1}
        self._reader = object()
        self._acq_path = "/nowhere"

    def run_operator(self, key, **kw):
        self.calls.append((key, kw))

    def say(self, text):
        self.said.append(text)

    def publish_qc_result(self, view, title):
        self.published.append((view, title))


def test_the_stitcher_panel_builds_and_offers_the_ported_controls(qapp):
    p = StitcherPanel(_Host())
    assert p.register_cb is not None
    assert p.reg_channel_combo.count() == 2
    assert p.blend_spin.value() == STITCH_DEFAULTS["blend_px"]
    assert p.rel_spin.value() == STITCH_DEFAULTS["outlier_rel_pct"]
    assert p.abs_spin.value() == STITCH_DEFAULTS["outlier_abs_px"]


def test_the_panel_does_not_carry_its_own_scope(qapp):
    """Scope belongs to the RUN; one representation, owned by pane 1's selector."""
    p = StitcherPanel(_Host())
    assert not hasattr(p, "scope_combo")


def test_the_stitcher_panel_is_parameters_only(qapp):
    """One flow (Julio, 2026-08-25): the panel carries NO run button and NO save checkbox -
    the view's operators row (Preview / Run on plate) launches every run and reads this
    panel through kwargs()."""
    host = _Host()
    p = StitcherPanel(host)
    assert not hasattr(p, "run_btn"), "the panel's own run button is back"
    assert not hasattr(p, "save_cb"), "the panel's save checkbox is back"
    p.register_cb.setChecked(False)
    p.blend_spin.setValue(64)
    kw = p.kwargs()
    assert kw["register"] is False
    assert kw["blend_px"] == 64
    assert host.calls == [], "building/reading the panel must launch nothing"


def test_turning_registration_off_disables_the_registration_only_controls(qapp):
    """A knob that provably does nothing must not look adjustable."""
    p = StitcherPanel(_Host())
    p.register_cb.setChecked(False)
    assert not p.rel_spin.isEnabled()
    assert not p.abs_spin.isEnabled()
    assert not p.reg_channel_combo.isEnabled()
    assert p.blend_spin.isEnabled()                  # fusion still feathers
    p.register_cb.setChecked(True)
    assert p.rel_spin.isEnabled()


def test_a_labels_z_operator_says_why_before_any_run(qapp, blob_operator):
    """A labels z-operator cannot be fused; the panel SAYS so the moment it is chosen (the
    launch-time guard is stitch_region's own refusal)."""
    host = _Host()
    p = StitcherPanel(host)
    p.z_operator_combo.setCurrentText(blob_operator)
    assert host.said and "label" in host.said[-1].lower()
    p.z_operator_combo.setCurrentText("mip")
    assert host.said[-1] == ""                  # the refusal clears with a legal choice


def test_an_intensity_z_operator_raises_no_refusal(qapp):
    """The refusal follows the ENGINE, not a guard the engine outgrew."""
    host = _Host()
    p = StitcherPanel(host)
    p.z_operator_combo.setCurrentText("decon")
    assert host.said[-1] == ""


def test_keep_every_plane_is_offered_and_spells_z_operator_none(qapp):
    """The shelved `keepz` identity's replacement: the combo's label maps to z_operator=None —
    every acquired plane fused unchanged — and never reaches the registry as a name.
    `z_operator_choice` is the one mapping, read by the plate's `operator_kwargs_for`."""
    from squidxplorer._op_panels import KEEP_EVERY_PLANE, z_operator_choice

    host = _Host()
    p = StitcherPanel(host)
    labels = [p.z_operator_combo.itemText(i) for i in range(p.z_operator_combo.count())]
    assert KEEP_EVERY_PLANE in labels
    assert "keepz" not in labels
    assert z_operator_choice(KEEP_EVERY_PLANE) is None
    assert z_operator_choice("mip") == "mip"


def test_the_decon_panel_starts_at_the_qc_start_iteration_count(qapp):
    from squidxplorer._decon import QC_START_ITERATIONS

    p = DeconQCPanel(_Host())
    assert p.iter_spin.value() == QC_START_ITERATIONS


def test_the_decon_panel_add_one_button_advances_by_exactly_one(qapp):
    p = DeconQCPanel(_Host())
    p.iter_spin.setValue(2)
    p.plus_btn.click()
    assert p.iter_spin.value() == 3


def test_the_decon_panel_shutdown_joins_a_running_worker(qapp):
    """Closing the tab mid-run must JOIN the RL thread rather than drop the last reference to
    a running QThread (which aborts the interpreter); the teardown path calls ``shutdown()``."""
    import threading

    from squidxplorer._op_panels import _DeconQCWorker

    assert hasattr(DeconQCPanel, "shutdown"), "the teardown path calls shutdown(), not stop()"

    started = threading.Event()

    class _SlowWorker(_DeconQCWorker):
        def __init__(self):
            super().__init__(None, "A1", 0, "c0", 1, False, 8, 8)

        def run(self):                       # stand in for an in-flight RL run
            started.set()
            while not self.isInterruptionRequested():
                self.msleep(10)

    p = DeconQCPanel(_Host())
    p._worker = _SlowWorker()
    p._worker.start()
    assert started.wait(2.0) and p._worker.isRunning()

    p.shutdown()                             # must interrupt + wait(), never abort

    assert p._worker is None                 # the worker was reaped, not orphaned


def test_the_result_view_carries_no_turbo_picture(qapp):
    """The turbo composite panes are GONE (Julio, 2026-08-25: "The turbo colormap preview
    makes no sense. remove it") — the preview is the in-view data layer under the channel's
    own colormap. The stepper, caption and metric survive them."""
    view = DeconQCResultView("A1/0/c0")
    assert not hasattr(view, "image_label"), "the turbo picture pane is back"
    assert not hasattr(view, "crosshair_label"), "the crosshair chrome is back"
    import squidxplorer._decon_qc as decon_qc
    import squidxplorer._op_panels as op_panels
    assert not hasattr(decon_qc, "turbo_rgb"), "_decon_qc still ships the turbo mapper"
    assert not hasattr(decon_qc, "qc_composite"), "_decon_qc still ships the composite"
    assert "turbo_rgb" not in open(op_panels.__file__).read()
    view.show_iteration(3, 0.31, "improving", "still tightening")
    assert "ITERATION 3 of 3" in view.caption_label.text()
    view.close()


def test_the_worker_hands_the_volume_through_for_the_data_preview(qapp):
    """The RL volume must survive the worker — it IS the preview the view renders."""
    from squidxplorer._op_panels import QCFrame

    volume = np.zeros((5, 40, 40), dtype=np.float32)
    volume[2, 20, 20] = 1000.0
    panel = DeconQCPanel(_Host())
    panel._view = DeconQCResultView("A1/0/c0")
    frame = QCFrame(volume, (2, 20, 20), 8)

    panel._on_done(3, frame, 0.31)

    rec = panel._view.capture(3)
    assert rec is not None and rec["volume"] is not None, (
        "the view got a caption but no volume to preview")
    assert rec["centre"] == (2, 20, 20)
    assert rec["view_half"] == 8
    panel._view.close()


def test_the_result_view_keeps_every_iteration_so_they_can_be_compared(qapp):
    view = DeconQCResultView("A1/0/c0")
    view.show_iteration(2, 0.40, "first", "")
    view.show_iteration(3, 0.31, "improving", "")
    assert [k for k, _ in view.history] == [2, 3]


# ---------------------------------------------------------------------------------------
# the controls ported from maragall/stitcher, and their kwargs
# ---------------------------------------------------------------------------------------

def test_every_kwarg_the_panel_emits_is_a_real_stitch_region_parameter():
    import inspect

    from squidxplorer._stitch import stitch_region

    kw = stitch_operator_kwargs(
        register=True, registration_channel=0, channels=None, blend_px=64,
        outlier_rel_pct=50, outlier_abs_px=2, correct_distortion=True, registration_t=3)
    allowed = set(inspect.signature(stitch_region).parameters)
    assert set(kw) <= allowed, set(kw) - allowed


def test_auto_blend_is_spelled_None_all_the_way_down():
    """``stitch_region`` measures the overlap when blend_px is None; sending the spin's stale
    number instead would silently ignore the checkbox."""
    kw = stitch_operator_kwargs(
        register=True, registration_channel=0, channels=None, blend_px=999,
        outlier_rel_pct=50, outlier_abs_px=2, auto_blend=True)
    assert kw["blend_px"] is None


def test_auto_blend_skips_the_ramp_vs_tile_refusal():
    """The "ramp must fit inside the tile" check is about a number the user typed; with Auto
    on there is no such number yet."""
    kw = stitch_operator_kwargs(
        register=True, registration_channel=0, channels=None, blend_px=5000,
        outlier_rel_pct=50, outlier_abs_px=2, auto_blend=True, tile_px=2084)
    assert kw["blend_px"] is None


def test_distortion_and_timepoint_are_dropped_when_registration_is_off():
    """Both are registration-only; forwarding them with register=False would make
    stitch_region refuse a run the user could not see they had configured."""
    kw = stitch_operator_kwargs(
        register=False, registration_channel=None, channels=None, blend_px=64,
        outlier_rel_pct=50, outlier_abs_px=2, correct_distortion=True, registration_t=2)
    assert "correct_distortion" not in kw
    assert "registration_t" not in kw


def test_the_panel_offers_the_distortion_and_auto_blend_controls(qapp):
    p = StitcherPanel(_Host())
    assert p.distortion_cb is not None
    assert p.blend_auto_cb is not None
    assert p.distortion_cb.isChecked()          # ON by default (Julio, 2026-08-03)


def test_auto_blend_disables_the_manual_width_so_no_dead_number_is_shown(qapp):
    p = StitcherPanel(_Host())
    p.blend_auto_cb.setChecked(True)
    assert not p.blend_spin.isEnabled()
    p.blend_auto_cb.setChecked(False)
    assert p.blend_spin.isEnabled()


def test_the_distortion_checkbox_is_greyed_out_with_registration_off(qapp):
    """maragall/stitcher's own version of this checkbox is never read at all; ours must at
    least not look adjustable when it provably does nothing."""
    p = StitcherPanel(_Host())
    p.register_cb.setChecked(False)
    assert not p.distortion_cb.isEnabled()


def test_the_panel_s_distortion_choice_travels_to_the_operator(qapp):
    p = StitcherPanel(_Host())
    p.distortion_cb.setChecked(True)
    assert p.kwargs()["correct_distortion"] is True


def test_the_timepoint_spin_is_hidden_on_a_single_timepoint_acquisition(qapp):
    """A spin whose only legal value is 0 is furniture."""
    p = StitcherPanel(_Host())
    assert p.reg_t_spin.maximum() == 0


def test_the_stitch_defaults_are_read_off_the_declaration_not_mirrored_by_hand():
    """The panel's starting position IS the ``stitch`` registration's ``params=``; the two
    outlier thresholds are undeclared (see ``_stitch._STITCH_PARAMS``) and come from
    ``_stitch``'s own constants."""
    from squidxplorer._engine import operator_params
    from squidxplorer._stitch import _ABS_THRESH, _REL_THRESH

    declared = {p.name: p.default for p in operator_params("stitch")}
    assert STITCH_DEFAULTS["register"] == declared["register"]
    assert STITCH_DEFAULTS["registration_channel"] == declared["registration_channel"]
    assert STITCH_DEFAULTS["registration_t"] == declared["registration_t"]
    assert STITCH_DEFAULTS["outlier_rel_pct"] == int(round(_REL_THRESH * 100))
    assert STITCH_DEFAULTS["outlier_abs_px"] == int(round(_ABS_THRESH))


def test_the_declaration_states_stitch_region_s_own_defaults():
    """A declared default drifting from the signature would make the registered run and a
    direct ``stitch_region`` call two different pipelines. A None signature default states its
    fixed meaning concretely instead (registration_channel None = index 0,
    correct_illumination None = on) and is exempt here."""
    from inspect import signature

    from squidxplorer._engine import operator_params
    from squidxplorer._stitch import stitch_region

    sig = signature(stitch_region).parameters
    for p in operator_params("stitch"):
        want = sig[p.name].default
        if want is not None:
            assert p.default == want, f"{p.name}: declared {p.default!r}, signature {want!r}"


# =======================================================================================
# THE GENERIC PANEL: a declared Param becomes a widget
# =======================================================================================
# `_engine.Operator.params` is read by bind/CLI/_recipe but was not read by the GUI,
# so `spot`/`cellpose` declared four parameters each and none were reachable from a panel.
# These tests are over `squidxplorer._param_panel`, whose policy half is Qt-free.

from squidxplorer._engine import Param  # noqa: E402
from squidxplorer._param_panel import (  # noqa: E402
    WIDGET_KINDS,
    GenericOperatorPanel,
    panel_refusal,
    unsupported_params,
    widget_kind,
)


# -- the mapping rule -------------------------------------------------------------------

def test_the_widget_is_chosen_from_the_type_of_the_default():
    """``Param`` declares no type/range/widget hint, so the default's type is the only
    dispatch key there is."""
    assert widget_kind(2.0) == "decimal"
    assert widget_kind(30) == "spin"
    assert widget_kind(True) == "check"
    assert widget_kind("nuclei") == "text"


def test_a_bool_is_a_check_box_and_not_a_zero_one_spinner():
    """``bool`` is a subclass of ``int``, so an isinstance ladder in the wrong order would
    render every check box as a 0/1 spin — hence the exact-type lookup table."""
    assert widget_kind(True) == "check"
    assert widget_kind(False) == "check"
    assert WIDGET_KINDS[bool] != WIDGET_KINDS[int]


def test_a_default_this_panel_cannot_draw_is_named_rather_than_guessed():
    """No silent fallback to a text box: a guessed widget is how a typed value becomes a
    value the run never receives."""
    assert widget_kind(None) is None
    assert widget_kind((1, 2)) is None
    bad = unsupported_params((Param("sigma", 2.0), Param("mask", None)))
    assert bad == [("mask", "NoneType")]


# -- the refusal ------------------------------------------------------------------------

def test_a_chain_expression_key_is_refused_naming_the_removal():
    why = panel_refusal("demo + blob")
    assert why is not None and "chaining was removed" in why

def test_a_parameterised_operator_is_not_refused(blob_operator):
    assert panel_refusal(blob_operator) is None


def test_a_region_operator_that_declares_no_params_is_refused_for_that_reason():
    """A param-less region operator has nothing for a form to show — the refusal names that.

    ``coordinate`` used to be the example; it now DECLARES z_operator/correct_illumination and
    gets a generated panel like every declaring operator, so the refusal is pinned on a bare
    runtime registration instead (the registry snapshot fixture restores the table)."""
    from squidxplorer import add_region_operator

    add_region_operator("bare_region_op_for_panels", lambda reader, region, fovs: None)
    why = panel_refusal("bare_region_op_for_panels")
    assert why and "declares no params" in why

    assert panel_refusal("register") is None, (
        "register declares params; a generated panel must serve it, not a refusal")


def test_a_key_that_is_not_an_operator_is_refused_by_name():
    """A key the registry never saw is refused with the registry's own sentence."""
    why = panel_refusal("not_an_operator")
    assert why and "not_an_operator" in why


def test_an_undrawable_parameter_refuses_the_whole_panel_naming_the_parameter():
    """A panel that silently omitted the one parameter it could not draw would run that
    parameter at its default while every other control implied the form was complete."""
    from squidxplorer import add_operator

    def _factory(**kwargs):
        def _op(planes):
            return next(iter(planes))
        return _op

    name = "panel_test_undrawable"
    add_operator(name, _factory, params=(Param("sigma_px", 2.0), Param("mask", None)))
    try:
        why = panel_refusal(name)
        assert why and "mask" in why and "NoneType" in why
    finally:
        from squidxplorer._engine import _OPERATORS
        _OPERATORS.pop(name, None)


# -- the Qt half ------------------------------------------------------------------------

def test_the_panel_builds_one_widget_per_declared_parameter(qapp, blob_operator):
    from squidxplorer._engine import operator_params

    p = GenericOperatorPanel(_Host(), blob_operator)
    assert sorted(p.widgets) == sorted(param.name for param in operator_params(blob_operator))
    assert len(p.widgets) == 3


def test_each_widget_starts_at_the_declared_default(qapp, blob_operator):
    """An untouched panel must launch what the operator ships with — same rule
    ``STITCH_DEFAULTS`` is held to."""
    from squidxplorer._engine import operator_params

    p = GenericOperatorPanel(_Host(), blob_operator)
    declared = {param.name: param.default for param in operator_params(blob_operator)}
    assert p.kwargs() == declared


def test_the_blurb_becomes_the_tooltip(qapp, blob_operator):
    from squidxplorer._engine import operator_params

    blurbs = {param.name: param.blurb for param in operator_params(blob_operator)}
    p = GenericOperatorPanel(_Host(), blob_operator)
    assert p.widgets, "the panel drew no widgets"
    for name, widget in p.widgets.items():
        assert blurbs[name], f"{name} declares no blurb, so the tooltip claim is untested"
        assert widget.toolTip() == blurbs[name]


def test_a_value_read_back_keeps_the_declared_type(qapp, blob_operator):
    """A ``min_area_px`` arriving as 30.0 where 30 was declared would survive all the way to
    a comparison against an integer pixel count."""
    p = GenericOperatorPanel(_Host(), blob_operator)
    p.widgets["min_area_px"].setValue(400)
    kwargs = p.kwargs()
    assert kwargs["min_area_px"] == 400 and isinstance(kwargs["min_area_px"], int)
    assert isinstance(kwargs["sigma_px"], float)
    assert isinstance(kwargs["split_touching"], bool)


def test_the_widget_s_value_is_what_kwargs_hands_the_run(qapp, blob_operator):
    """The operators row reads this panel through ``kwargs()`` (`operator_kwargs_for`), so
    the widget's value IS what a run gets."""
    p = GenericOperatorPanel(_Host(), blob_operator)
    p.widgets["min_area_px"].setValue(400)
    p.widgets["split_touching"].setChecked(False)
    kw = p.kwargs()
    assert kw["min_area_px"] == 400
    assert kw["split_touching"] is False


def test_every_kwarg_the_panel_emits_is_a_parameter_the_operator_accepts(qapp,
                                                                          blob_operator):
    """The panel's output must survive ``Operator.bind``, which refuses an unknown name loud."""
    from squidxplorer import bind_operator

    p = GenericOperatorPanel(_Host(), blob_operator)
    p.widgets["min_area_px"].setValue(80)
    bind_operator(blob_operator, p.kwargs())   # raises if wrong


def test_the_generic_panel_carries_no_run_buttons(qapp, blob_operator):
    """One flow (Julio, 2026-08-25): parameters only. Preview and Run on plate live in the
    view's operators row; the old per-panel preview spinner and save button are gone."""
    from qtpy.QtWidgets import QPushButton

    p = GenericOperatorPanel(_Host(), blob_operator)
    assert not hasattr(p, "run_btn") and not hasattr(p, "save_btn")
    assert not hasattr(p, "wells_spin")
    assert [b.text() for b in p.findChildren(QPushButton)] == []


def test_an_operator_with_no_parameters_still_builds_and_says_so(qapp):
    """``mip`` declares nothing; a panel that refused would make "no parameters" and "unknown
    operator" look identical."""
    p = GenericOperatorPanel(_Host(), "mip")
    assert p.widgets == {}
    assert p.kwargs() == {}


# =======================================================================================
# THE QC SWEEP STEPPER + the session optics row (2026-08-24)
# =======================================================================================
# Julio (2026-08-24): step "iteration by iteration"; (2026-08-25): "The turbo colormap
# preview makes no sense. remove it." The sweep captures every iteration of ONE solve; the
# view steps them; 'use k iterations' writes k into the run's ONE iterations parameter; the
# preview is the in-view data layer under the channel's own colormap.

@pytest.fixture()
def clean_decon_session():
    """The panel installs session NI/NA process-wide on purpose; tests must not leak it."""
    from squidxplorer._decon import set_session_na, set_session_ni

    yield
    set_session_ni(None)
    set_session_na(None)


def _sweep_into(view, ks=(1, 2, 3)):
    """Feed *view* one distinct captured iteration per k, the way the worker's done signal does."""
    volumes = {}
    for k in ks:
        volume = np.zeros((5, 20, 20), dtype=np.float32)
        volume[2, 10, 10] = 100.0 * k               # genuinely different pixels per iteration
        volumes[k] = volume
        view.show_iteration(k, 0.5 - 0.1 * k,
                            "improving", "still tightening", volume=volume,
                            centre=(2, 10, 10), delta=None if k == 1 else float(k))
    return volumes


def test_the_view_steps_iteration_by_iteration_without_a_re_solve(qapp, clean_decon_session):
    view = DeconQCResultView("A1/0/c0")
    volumes = _sweep_into(view)

    assert "ITERATION 3 of 3" in view.caption_label.text()
    view.prev_btn.click()
    assert "ITERATION 2 of 3" in view.caption_label.text()
    assert np.array_equal(view.capture(view._shown_k)["volume"], volumes[2]), (
        "stepping back did not land on iteration 2's captured volume")
    view.iter_slider.setValue(1)
    assert "ITERATION 1 of 3" in view.caption_label.text()
    view.next_btn.click()
    assert "ITERATION 2 of 3" in view.caption_label.text()
    assert [k for k, _ in view.history] == [1, 2, 3], "stepping polluted the history"
    view.close()


def test_the_per_step_change_is_shown_beside_the_ratio(qapp, clean_decon_session):
    """mean |Δ| vs the previous iteration is the honest 'is it still moving' number; the first
    iteration has no previous and must not invent one."""
    view = DeconQCResultView("A1/0/c0")
    _sweep_into(view, ks=(1, 2))
    assert "mean |Δ| vs k-1: 2" in view.caption_label.text()
    view.prev_btn.click()
    assert "mean |Δ|" not in view.caption_label.text(), "iteration 1 shows a delta against nothing"
    view.close()


def test_use_k_iterations_writes_the_displayed_count_into_the_run(qapp, clean_decon_session):
    """THE point of the preview: the displayed k lands in the panel's run-iterations control,
    which is what kwargs() -> operator_kwargs_for feeds every decon run. One source of truth."""
    from squidxplorer._decon import DEFAULT_ITERATIONS

    panel = DeconQCPanel(_Host())
    assert panel.kwargs() == {"iterations": DEFAULT_ITERATIONS}
    view = DeconQCResultView("A1/0/c0")
    view.useIterations.connect(panel._adopt_iterations)
    _sweep_into(view)
    view.prev_btn.click()                          # judge by eye: iteration 2 looks right
    assert "Use 2 iteration" in view.use_btn.text()
    view.use_btn.click()
    assert panel.run_iter_spin.value() == 2
    assert panel.kwargs() == {"iterations": 2}
    assert any("2 iteration" in s for s in panel.host.said)
    view.close()


def test_the_result_view_lives_inline_in_the_panel_never_published(qapp, clean_decon_session):
    """The whole choose-the-iteration loop travels WITH the panel (into the operator dock);
    publish_qc_result — the plate-tab seam — is deliberately not called any more."""
    host = _Host()
    panel = DeconQCPanel(host)
    panel.run()                                    # dataset is bogus; the view exists anyway
    try:
        assert panel._view is not None
        assert panel.isAncestorOf(panel._view), "the view is not inside the panel"
        assert host.published == [], "the panel still throws the picture to another window"
    finally:
        panel.shutdown()


def test_the_panel_assumes_air_and_a_pick_reaches_the_session(qapp, clean_decon_session):
    from squidxplorer._decon import session_ni, set_session_ni

    set_session_ni(None)
    p = DeconQCPanel(_Host())
    assert p.ni_combo.currentText() == "1.000 (air)", "value first, medium in parentheses"
    assert session_ni() == pytest.approx(1.000), "opening the panel did not install air"
    p.ni_combo.setCurrentIndex(1)
    assert p.ni_combo.currentText() == "1.333 (water)"
    assert session_ni() == pytest.approx(1.333)


def test_a_rebuilt_panel_keeps_the_session_medium(qapp, clean_decon_session):
    """Session-scoped means a re-opened panel shows the medium already chosen, not air again."""
    from squidxplorer._decon import set_session_ni

    set_session_ni(1.515)
    p = DeconQCPanel(_Host())
    assert p.ni_combo.currentText() == "1.515 (oil)"


def test_an_impossible_na_is_flagged_by_name_in_the_panel(qapp, clean_decon_session):
    p = DeconQCPanel(_Host())                      # air installed
    p.na_spin.setValue(1.40)
    assert any("impossible in air" in s for s in p.host.said), (
        "NA 1.40 under air went unflagged")


def test_the_worker_sweeps_a_real_stack_emitting_every_iteration(qapp, clean_decon_session,
                                                                 squid_dataset, monkeypatch):
    """End to end on the real fixture: iterations=2 must deliver k=1 AND k=2 (each with its
    volume and a delta from k >= 2) and then sweep_done — off the Qt thread."""
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("SQUIDXPLORER_DECON_DEVICE", "cpu")
    from squidxplorer import open_reader
    from squidxplorer._op_panels import _DeconQCWorker

    root, _ = squid_dataset
    meta = open_reader(str(root)).metadata
    region, channel = meta["regions"][0], meta["channels"][0]["name"]
    got, finished, failures = [], [], []
    worker = _DeconQCWorker(str(root), region, 0, channel, 2, False, 8, 8)
    worker.done.connect(lambda k, frame, ratio: got.append((k, frame, ratio)))
    worker.sweep_done.connect(lambda n: finished.append(n))
    worker.failed.connect(lambda m: failures.append(m))
    worker.start()
    import time
    deadline = time.monotonic() + 60
    while not (finished or failures) and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    worker.wait(5000)
    assert not failures, f"the sweep failed: {failures}"
    assert finished == [2]
    assert [k for k, _, _ in got] == [1, 2]
    for k, frame, ratio in got:
        assert frame.volume is not None and frame.centre is not None
        assert frame.delta is not None, "every capture carries its mean |Δ|"
    assert not np.array_equal(got[0][1].volume, got[1][1].volume), (
        "iteration 1 and 2 delivered the same volume")


# ── the preview-vs-run contract and the final PSF parameters (UI feedback 2026-08-24) ─────────


def test_the_panel_states_the_preview_channel_vs_run_contract(qapp, clean_decon_session):
    """Julio read a one-channel preview as "decon was running on only one channel": the
    channel combo is the PREVIEW picker, and the caption says the run covers every channel."""
    p = DeconQCPanel(_Host(channels=("c0", "c1")))
    assert "preview: c0 only" in p.preview_note.text()
    assert "every channel" in p.preview_note.text()
    assert "own PSF" in p.preview_note.text()
    p.channel_combo.setCurrentIndex(1)
    assert "preview: c1 only" in p.preview_note.text(), (
        "the caption did not follow the preview channel")


def _fixed_optics_panel(host, wavelength_by=None):
    """A panel whose recorded optics are injected, so the effective line is deterministic."""
    from squidxplorer._decon import OpticsParams

    panel = DeconQCPanel(host)

    def _recorded():
        ch = panel.channel_combo.currentText()
        wl = (wavelength_by or {}).get(ch, 0.525)
        return OpticsParams(na=0.80, wavelength_um=wl, dxy_um=0.752, dz_um=1.5, nz=10), ""

    panel._recorded_optics = _recorded
    return panel


def test_the_effective_line_shows_what_the_solve_will_use(qapp, clean_decon_session):
    """Item 4: the FINAL values the PSF is built from — session NI/NA applied, medium named,
    magnification derived from the sensor pixel — live-updating with the optics row."""
    panel = _fixed_optics_panel(_Host())
    panel._sensor_pixel_um = lambda: 7.52
    panel._refresh_optics_note()
    line = panel.effective_note.text()
    assert "the solve will use" in line
    assert "NA 0.80" in line and "air (ni 1.000)" in line
    assert "emission 0.525" in line and "(c0)" in line
    assert "pixel 0.752" in line and "dz 1.50" in line and "nz 10" in line
    assert "magnification 10.0x" in line, f"no derived magnification in {line!r}"

    panel.ni_combo.setCurrentIndex(1)              # water: the line must follow the session
    assert "water (ni 1.333)" in panel.effective_note.text()
    panel.na_spin.setValue(0.95)
    assert "NA 0.95" in panel.effective_note.text(), "the session NA override is not shown"


def test_the_effective_line_follows_the_preview_channel_s_own_wavelength(qapp,
                                                                         clean_decon_session):
    """Every channel gets its OWN PSF: switching the preview channel must switch the shown
    emission wavelength, or the line describes another channel's solve."""
    panel = _fixed_optics_panel(_Host(channels=("c0", "c1")),
                                wavelength_by={"c0": 0.525, "c1": 0.670})
    panel._refresh_optics_note()
    assert "emission 0.525" in panel.effective_note.text()
    panel.channel_combo.setCurrentIndex(1)
    assert "emission 0.670" in panel.effective_note.text()


def test_an_impossible_session_na_turns_the_effective_line_into_the_refusal(qapp,
                                                                            clean_decon_session):
    panel = _fixed_optics_panel(_Host())
    panel.na_spin.setValue(1.40)                   # impossible in air
    assert "the solve will refuse" in panel.effective_note.text()


def test_set_param_writes_the_run_iterations_and_refuses_unknown_names(qapp,
                                                                       clean_decon_session):
    panel = DeconQCPanel(_Host())
    assert panel.set_param("iterations", 9) is None
    assert panel.kwargs() == {"iterations": 9}
    why = panel.set_param("blend_px", 3)
    assert why and "blend_px" in why


# ── the sweep's data preview lands in a view as a real layer (item 1) ─────────────────────────


def _preview_rig(channels=("c0",)):
    """(panel, mosaic, view): a host with a manager whose one view shows A1 with positions."""
    from napari.components import ViewerModel

    from squidxplorer._napari_view import MosaicLayers

    host = _Host(channels=channels)
    meta = dict(host._meta)
    meta["fov_positions_um"] = {("A1", 0): (100.0, 200.0), ("A1", 1): (356.0, 200.0),
                                ("A2", 0): (900.0, 200.0), ("A2", 1): (1156.0, 200.0)}
    mosaic = MosaicLayers(ViewerModel())
    view = type("V", (), {})()
    view._regions = ["A1", "A2"]
    view._meta = meta
    view._pane = type("P", (), {"mosaic": mosaic})()
    mgr = type("M", (), {"active_view": lambda self: view,
                         "windows": lambda self: [view]})()
    host._viewer_manager = mgr
    panel = DeconQCPanel(host)
    return panel, mosaic, view


def test_the_displayed_iteration_lands_in_a_view_at_its_stage_position(qapp,
                                                                       clean_decon_session):
    """Item 1: the sweep is judged on ACTUAL data at real size — the displayed capture lands
    as a real DATA layer (the channel's own colormap, never turbo) in a view showing that
    region, placed at the crop's own stage footprint, and stepping the slider swaps the SAME
    layer's pixels (no re-solve)."""
    panel, mosaic, _view = _preview_rig()
    panel._sweep_at = ("A1", 0, "c0")
    panel._view = DeconQCResultView("A1/0/c0")
    panel._view.iterationDisplayed.connect(panel._push_view_preview)

    vol1 = np.zeros((2, 16, 16), np.float32); vol1[1, 8, 8] = 100.0
    panel._view.show_iteration(1, 0.5, "first", "v",
                               volume=vol1, centre=(1, 8, 8), fov_origin=(6, 4))

    layer = mosaic.find(DeconQCPanel.PREVIEW_OP, "c0")
    assert layer is not None, "the displayed capture never reached the view"
    assert tuple(np.asarray(layer.data).shape) == (2, 16, 16), "the preview lost its z"
    # FOV 0's top-left is the region origin (100, 200); the crop sits +4 px x, +6 px y at 1 um/px.
    assert tuple(float(v) for v in layer.translate[-2:]) == (206.0, 104.0)
    assert tuple(float(v) for v in layer.scale) == (1.5, 1.0, 1.0), (
        "the crop is not placed at the acquisition's own pitch and z step")
    assert getattr(layer.colormap, "name", "") != "turbo", (
        "the preview layer still renders in turbo; it must wear the channel's own colormap")

    vol2 = np.zeros((2, 16, 16), np.float32); vol2[1, 8, 8] = 300.0
    panel._view.show_iteration(2, 0.4, "improving", "v",
                               volume=vol2, centre=(1, 8, 8), fov_origin=(6, 4), delta=1.0)
    assert mosaic.find(DeconQCPanel.PREVIEW_OP, "c0") is layer, (
        "stepping created a second layer instead of updating the preview")
    assert float(np.asarray(layer.data).max()) == 300.0
    panel._view.iter_slider.setValue(1)            # step BACK: the k=1 capture repaints
    assert float(np.asarray(layer.data).max()) == 100.0

    panel._view.close()


def test_run_wires_the_stepper_to_the_view_preview(qapp, clean_decon_session, monkeypatch):
    """The production wiring: run() itself connects iterationDisplayed to the preview push."""
    from squidxplorer._op_panels import _DeconQCWorker

    monkeypatch.setattr(_DeconQCWorker, "start", lambda self: None)
    panel, mosaic, _view = _preview_rig()
    panel.run()
    try:
        vol = np.zeros((2, 16, 16), np.float32); vol[1, 8, 8] = 50.0
        panel._view.show_iteration(1, 0.5, "first", "v",
                                   volume=vol, centre=(1, 8, 8), fov_origin=(0, 0))
        assert mosaic.find(DeconQCPanel.PREVIEW_OP, "c0") is not None, (
            "run() did not wire the stepper to the in-view data preview")
    finally:
        panel.shutdown()


def test_a_sweep_with_no_view_over_the_region_still_shows_in_the_panel(qapp,
                                                                       clean_decon_session):
    """No view over the region: the push is a quiet no-op, never an error — the panel's own
    caption and metric still describe the sweep."""
    panel = DeconQCPanel(_Host())                  # host has no _viewer_manager at all
    panel._sweep_at = ("A1", 0, "c0")
    panel._view = DeconQCResultView("A1/0/c0")
    vol = np.zeros((2, 16, 16), np.float32)
    panel._view.show_iteration(1, 0.5, "first", "v",
                               volume=vol, centre=(1, 8, 8), fov_origin=(0, 0))
    panel._push_view_preview(1)                    # must not raise
    panel._view.close()



# ── the text diet + parameter hiding (Julio, 2026-08-25) ──────────────────────────────────────
# "There is so much text in the operator UI. As if it was a book - that's crazy lol."
# "There are so many parameters that should be default and our life science user should not
# want to see them." "recorded NA should print the value to it's side."


def test_the_decon_headline_is_iterations_ni_na_and_the_rest_hides(qapp, clean_decon_session):
    """The panel shows the headline knobs; where-to-measure and the sweep knobs live behind
    a 'more' disclosure, the PSF lines behind their own toggle - both closed by default."""
    p = DeconQCPanel(_Host())
    assert p._more.isHidden(), "the advanced knobs are on screen by default"
    assert p._psf.isHidden(), "the PSF detail is on screen by default"
    # the advanced knobs really moved INSIDE the disclosures
    assert p.region_combo in p._more.findChildren(type(p.region_combo))
    assert p.optics_note in p._psf.findChildren(type(p.optics_note))
    # the headline stays out of them
    assert p.run_iter_spin not in p._more.findChildren(type(p.run_iter_spin))
    assert p.ni_combo not in p._more.findChildren(type(p.ni_combo))
    p.more_btn.click()
    assert not p._more.isHidden()
    p.psf_btn.click()
    assert not p._psf.isHidden()


def test_ni_offers_custom_and_a_typed_value_reaches_the_session(qapp, clean_decon_session):
    from squidxplorer._decon import session_ni

    p = DeconQCPanel(_Host())
    labels = [p.ni_combo.itemText(i) for i in range(p.ni_combo.count())]
    assert "custom" in labels, "the NI dropdown offers no custom entry"
    assert p.ni_custom.isHidden(), "the custom NI field shows while a standard medium is picked"
    p.ni_combo.setCurrentText("custom")
    assert not p.ni_custom.isHidden()
    p.ni_custom.setValue(1.38)
    assert session_ni() == pytest.approx(1.38), "the typed NI never reached the session"
    p.ni_combo.setCurrentText("1.333 (water)")
    assert p.ni_custom.isHidden()
    assert session_ni() == pytest.approx(1.333)


def test_recorded_na_prints_its_value_beside_the_control(qapp, clean_decon_session):
    panel = _fixed_optics_panel(_Host())
    panel._refresh_optics_note()
    assert "recorded NA: 0.80" in panel.na_recorded.text(), (
        f"the recorded NA is not printed beside the control: {panel.na_recorded.text()!r}")


def test_magnification_stays_geometric_when_na_or_ni_change(qapp, clean_decon_session):
    """mag = sensor / dxy. NA and ni shape the PSF, not the geometry; a magnification that
    moved with them would be a fake dependence."""
    panel = _fixed_optics_panel(_Host())
    panel._sensor_pixel_um = lambda: 7.52
    panel._refresh_optics_note()
    assert "magnification 10.0x" in panel.effective_note.text()
    panel.na_spin.setValue(0.95)
    assert "magnification 10.0x" in panel.effective_note.text(), (
        "the shown magnification moved with NA")
    panel.ni_combo.setCurrentText("1.333 (water)")
    assert "magnification 10.0x" in panel.effective_note.text(), (
        "the shown magnification moved with ni")


def test_the_decon_panel_carries_no_book_of_text(qapp, clean_decon_session):
    """Every visible label is a line, not a paragraph (tooltips may keep a sentence; a
    refusal quoting the loader's own error is exempt, so the panel gets readable optics)."""
    from qtpy.QtWidgets import QLabel

    p = _fixed_optics_panel(_Host())
    p._sensor_pixel_um = lambda: 7.52
    p._refresh_optics_note()
    long_ones = [lab.text() for lab in p.findChildren(QLabel)
                 if len(lab.text()) > 200]
    assert not long_ones, f"paragraph-length label(s) in the panel: {long_ones}"
