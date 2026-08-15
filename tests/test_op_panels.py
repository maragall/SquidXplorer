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

def test_a_labels_operator_is_refused_with_a_sentence_naming_the_way_out():
    """``stitch_region`` raises for a labels operator; the panel asks the same registry first
    rather than let the user discover it after a multi-minute run."""
    why = stitch_refusal("cellpose")
    assert why is not None
    assert "cellpose" in why
    assert "label" in why.lower()
    assert "per FOV" in why or "intensity" in why    # it must say what to do instead


def test_a_plane_op_is_no_longer_refused():
    """Per-plane fusion made a plane-op stitchable; a pre-check outliving that change would be
    the engine's answer, wrong, delivered with authority."""
    assert stitch_refusal("decon") is None
    assert stitch_refusal("bgsub") is None
    assert stitch_refusal("flatfield") is None


def test_a_z_reducer_is_not_refused():
    assert stitch_refusal("mip") is None
    assert stitch_refusal("decon3d") is None


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


def test_the_run_leaves_scope_unresolved_so_the_run_selector_owns_it(qapp):
    """regions=None is UNSCOPED, not "the whole plate": run_operator resolves it against the
    LIVE selection, since a panel built once and cached would otherwise capture a stale one."""
    host = _Host()
    p = StitcherPanel(host)
    p.run_btn.click()
    assert host.calls[0][1]["regions"] is None


def test_the_stitcher_run_button_launches_the_operator_with_the_panel_s_kwargs(qapp):
    host = _Host()
    p = StitcherPanel(host)
    p.register_cb.setChecked(False)
    p.blend_spin.setValue(64)
    p.run_btn.click()
    assert len(host.calls) == 1
    key, kw = host.calls[0]
    assert key == "stitch"
    assert kw["operator_kwargs"]["register"] is False
    assert kw["operator_kwargs"]["blend_px"] == 64
    assert kw["save"] is False                       # tuning a fusion run is a preview


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


def test_a_labels_z_operator_disables_the_run_button_and_says_why(qapp):
    host = _Host()
    p = StitcherPanel(host)
    p.z_operator_combo.setCurrentText("cellpose")
    assert not p.run_btn.isEnabled()
    assert host.said and "label" in host.said[-1].lower()
    p.z_operator_combo.setCurrentText("mip")
    assert p.run_btn.isEnabled()


def test_a_plane_op_z_operator_leaves_the_run_button_enabled(qapp):
    """The button follows the ENGINE, not a guard the engine outgrew."""
    host = _Host()
    p = StitcherPanel(host)
    p.z_operator_combo.setCurrentText("decon")
    assert p.run_btn.isEnabled()


def test_the_run_handler_itself_refuses_labels_not_just_the_disabled_button(qapp):
    """Exercises the SECOND defence: clicking a disabled button never enters the handler, so
    the guard inside ``_run`` must refuse independently of the button's enabled state."""
    host = _Host()
    p = StitcherPanel(host)
    p.z_operator_combo.setCurrentText("cellpose")
    p.run_btn.setEnabled(True)                  # simulate reaching _run some other way
    p._run()
    assert host.calls == [], "the run must not start"
    assert "label" in host.said[-1].lower()


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


def test_the_result_view_renders_the_turbo_composite_at_the_composite_s_own_size(qapp):
    """Pane 3 renders the picture ``squidxplorer._decon_qc`` produced; it builds none of its own."""
    pytest.importorskip("matplotlib")
    from squidxplorer._decon_qc import qc_composite

    volume = np.zeros((5, 40, 40), dtype=np.float32)
    volume[2, 20, 20] = 1000.0
    composite = qc_composite(volume, (2, 20, 20), gap=2)
    view = DeconQCResultView("A1/0/c0")
    view.show_iteration(3, composite, 0.31, "improving", "still tightening")
    img = view.image_label.pixmap().toImage()
    assert (img.width(), img.height()) == (composite.shape[1], composite.shape[0])
    assert "3" in view.caption_label.text()


def _click(qapp, view, row, col):
    """A real mouse press at composite pixel (row, col), through ``mousePressEvent`` so the
    pixmap's centring offset inside the label is exercised, not assumed."""
    from qtpy.QtCore import QPoint
    from qtpy.QtTest import QTest

    label = view.image_label
    pm = label.pixmap()
    dx = max((label.width() - pm.width()) // 2, 0)
    dy = max((label.height() - pm.height()) // 2, 0)
    QTest.mouseClick(label, Qt.LeftButton, Qt.NoModifier, QPoint(dx + col, dy + row))
    qapp.processEvents()


def _qc_view(qapp, volume, centre, view_half=None):
    """A result view showing one iteration of *volume*, laid out and clickable."""
    from squidxplorer._decon_qc import qc_composite

    view = DeconQCResultView("A1/0/c0")
    view.show_iteration(3, qc_composite(volume, centre, view_half=view_half), 0.31,
                        "improving", "still tightening",
                        volume=volume, centre=centre, view_half=view_half)
    view.resize(400, 400)
    view.show()
    qapp.processEvents()
    return view


def test_clicking_the_picture_moves_the_crosshairs_and_re_cuts_the_strips(qapp):
    """A click re-sections the SAME volume through the clicked point; ``qc_composite`` already
    takes ``centre``, so no RL run happens here."""
    pytest.importorskip("matplotlib")
    volume = np.zeros((5, 40, 40), dtype=np.float32)
    volume[2, 20, 20] = 1000.0            # what the run centred on
    volume[2, 10, 10] = 700.0             # a second structure, off both current sections
    view = _qc_view(qapp, volume, (2, 20, 20))
    before = view._rgb.copy()

    _click(qapp, view, row=10, col=10)    # inside the x-y panel

    assert view._centre == (2, 10, 10), "the crosshairs did not move to the clicked voxel"
    assert not np.array_equal(view._rgb, before), (
        "the crosshairs moved but the strips were not re-cut — the picture is stale")
    assert "z=2" in view.crosshair_label.text() and "y=10" in view.crosshair_label.text()
    # The halo/core number was measured where the RUN put the crosshairs, so once they move by
    # hand the picture and the number are about different points and the view has to say so.
    assert "moved by hand" in view.crosshair_label.text()
    assert view.history == [(3, pytest.approx(0.31))], "a click is not another iteration"
    view.close()


def test_clicking_a_separator_band_moves_nothing(qapp):
    """A gap pixel points at no section; snapping to the nearest one would move the crosshairs
    somewhere the user did not click."""
    pytest.importorskip("matplotlib")
    volume = np.zeros((5, 40, 40), dtype=np.float32)
    volume[2, 20, 20] = 1000.0
    view = _qc_view(qapp, volume, (2, 20, 20))
    before = view._rgb.copy()

    _click(qapp, view, row=41, col=10)    # the horizontal separator (gap=2 at rows 40..41)

    assert view._centre == (2, 20, 20)
    assert np.array_equal(view._rgb, before)
    view.close()


def test_a_view_shown_without_its_volume_is_simply_not_clickable(qapp):
    """The three-argument show_iteration must not raise on a click: there is nothing to re-slice."""
    pytest.importorskip("matplotlib")
    from squidxplorer._decon_qc import qc_composite

    volume = np.zeros((5, 40, 40), dtype=np.float32)
    volume[2, 20, 20] = 1000.0
    view = DeconQCResultView("A1/0/c0")
    view.show_iteration(3, qc_composite(volume, (2, 20, 20)), 0.31, "improving", "")
    view.resize(400, 400)
    view.show()
    qapp.processEvents()
    before = view._rgb.copy()

    _click(qapp, view, row=10, col=10)

    assert view._centre is None
    assert np.array_equal(view._rgb, before)
    view.close()


def test_the_worker_hands_the_volume_through_so_the_click_has_something_to_re_slice(qapp):
    """The RL volume must survive the worker — emitting the composite alone would leave nothing
    to cut a different section out of. Pinned via ``frame.volume``/``centre``/``view_half``."""
    pytest.importorskip("matplotlib")
    from squidxplorer._decon_qc import qc_composite
    from squidxplorer._op_panels import QCFrame

    volume = np.zeros((5, 40, 40), dtype=np.float32)
    volume[2, 20, 20] = 1000.0
    panel = DeconQCPanel(_Host())
    panel._view = DeconQCResultView("A1/0/c0")
    frame = QCFrame(qc_composite(volume, (2, 20, 20), view_half=8), volume, (2, 20, 20), 8)

    panel._on_done(3, frame, 0.31)

    assert panel._view._volume is not None, "the view got a picture but no volume to re-slice"
    assert panel._view._centre == (2, 20, 20)
    assert panel._view._view_half == 8
    panel._view.close()


def test_the_result_view_keeps_every_iteration_so_they_can_be_compared(qapp):
    pytest.importorskip("matplotlib")
    from squidxplorer._decon_qc import qc_composite

    volume = np.zeros((5, 20, 20), dtype=np.float32)
    volume[2, 10, 10] = 1000.0
    c = qc_composite(volume, (2, 10, 10), gap=2)
    view = DeconQCResultView("A1/0/c0")
    view.show_iteration(2, c, 0.40, "first", "")
    view.show_iteration(3, c, 0.31, "improving", "")
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
    host = _Host()
    p = StitcherPanel(host)
    p.distortion_cb.setChecked(True)
    p.run_btn.click()
    assert host.calls[0][1]["operator_kwargs"]["correct_distortion"] is True


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
    why = panel_refusal("bgsub + spot")
    assert why is not None and "chaining was removed" in why

def test_a_parameterised_operator_is_not_refused():
    assert panel_refusal("spot") is None
    # cellpose is the [segment] extra, not installed by `.[gui,test]`; on a clean venv it
    # correctly refuses "needs cellpose, which is not installed". Guarded rather than
    # deleted, because when the extra IS installed this pins that requires= being satisfied
    # still yields a panel.
    pytest.importorskip("cellpose")
    assert panel_refusal("cellpose") is None


def test_a_region_operator_that_declares_no_params_is_refused_for_that_reason():
    """A param-less region operator has nothing for a form to show — the refusal names that.

    ``coordinate`` used to be the example; it now DECLARES z_operator/correct_illumination and
    gets a generated panel like every declaring operator, so the refusal is pinned on a bare
    runtime registration instead (the registry snapshot fixture restores the table)."""
    from squidxplorer import add_region_operator

    add_region_operator("bare_region_op_for_panels", lambda reader, region, fovs: None)
    why = panel_refusal("bare_region_op_for_panels")
    assert why and "declares no params" in why

    assert panel_refusal("coordinate") is None, (
        "coordinate declares params now; a generated panel must serve it, not a refusal")


def test_a_key_that_is_not_an_operator_is_refused_by_name():
    """``minerva`` is a card, not an operator; refused with the registry's own sentence."""
    why = panel_refusal("minerva")
    assert why and "minerva" in why


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

def test_the_panel_builds_one_widget_per_declared_parameter(qapp):
    from squidxplorer._engine import operator_params

    p = GenericOperatorPanel(_Host(), "spot")
    assert sorted(p.widgets) == sorted(param.name for param in operator_params("spot"))
    assert len(p.widgets) == 4


def test_each_widget_starts_at_the_declared_default(qapp):
    """An untouched panel must launch what the operator ships with — same rule
    ``STITCH_DEFAULTS`` is held to."""
    from squidxplorer._engine import operator_params

    p = GenericOperatorPanel(_Host(), "spot")
    declared = {param.name: param.default for param in operator_params("spot")}
    assert p.kwargs() == declared


def test_the_blurb_becomes_the_tooltip(qapp):
    from squidxplorer._engine import operator_params

    blurbs = {param.name: param.blurb for param in operator_params("spot")}
    p = GenericOperatorPanel(_Host(), "spot")
    assert p.widgets, "the panel drew no widgets"
    for name, widget in p.widgets.items():
        assert blurbs[name], f"{name} declares no blurb, so the tooltip claim is untested"
        assert widget.toolTip() == blurbs[name]


def test_a_value_read_back_keeps_the_declared_type(qapp):
    """A ``min_area_px`` arriving as 30.0 where 30 was declared would survive all the way to
    a comparison against an integer pixel count."""
    p = GenericOperatorPanel(_Host(), "spot")
    p.widgets["min_area_px"].setValue(400)
    kwargs = p.kwargs()
    assert kwargs["min_area_px"] == 400 and isinstance(kwargs["min_area_px"], int)
    assert isinstance(kwargs["sigma_px"], float)
    assert isinstance(kwargs["split_touching"], bool)


def test_the_widget_s_value_travels_to_run_operator_through_operator_kwargs(qapp):
    """The SAME argument StitcherPanel uses, so this is an already-tested path."""
    host = _Host()
    p = GenericOperatorPanel(host, "spot")
    p.widgets["min_area_px"].setValue(400)
    p.widgets["split_touching"].setChecked(False)
    p.run_btn.click()
    key, kw = host.calls[0]
    assert key == "spot"
    assert kw["operator_kwargs"]["min_area_px"] == 400
    assert kw["operator_kwargs"]["split_touching"] is False
    assert kw["save"] is False


def test_every_kwarg_the_panel_emits_is_a_parameter_the_operator_accepts(qapp):
    """The panel's output must survive ``Operator.bind``, which refuses an unknown name loud."""
    from squidxplorer import bind_operator

    host = _Host()
    p = GenericOperatorPanel(host, "spot")
    p.widgets["min_area_px"].setValue(80)
    p.run_btn.click()
    bind_operator("spot", host.calls[0][1]["operator_kwargs"])   # raises if a name is wrong


def test_a_plane_op_is_offered_preview_only_and_the_choice_comes_off_consumes(qapp):
    """``spot`` keeps z at full depth, so there is no plate to save. Read off ``consumes``,
    never the name."""
    p = GenericOperatorPanel(_Host(), "spot")
    assert p._can_save is False
    assert p.save_btn is None                # not built at all, not built-and-hidden


def test_a_z_reducer_is_offered_the_save_run(qapp):
    p = GenericOperatorPanel(_Host(), "mip")
    assert p._can_save is True
    assert p.save_btn is not None and p.save_btn.parent() is not None


def test_an_operator_with_no_parameters_still_builds_and_says_so(qapp):
    """``mip`` declares nothing; a panel that refused would make "no parameters" and "unknown
    operator" look identical."""
    p = GenericOperatorPanel(_Host(), "mip")
    assert p.widgets == {}
    assert p.kwargs() == {}


