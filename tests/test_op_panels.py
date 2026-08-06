"""The two operator panels in PANE 1: their POLICY, separately from their pixels.

Julio: "Right now I'm blocked in testing the post-processing because Stitcher doesn't have
that maragall/Stitcher interface embedded in our top-left subpane."

So this covers the decisions those panels make -- which kwargs a registration/fusion run is
launched with, what the scope selector offers, and when an operator must REFUSE with a
sentence instead of running. All of it is pure functions over plain data, deliberately: a
control surface whose only test is "the widget constructed" is the kind of test this repo
has shipped dead before. The Qt half (that the widgets build and that the buttons are wired
to these functions) is at the bottom and runs offscreen.
"""

from __future__ import annotations

import numpy as np
import pytest

from squidmip._op_panels import (
    STITCH_DEFAULTS,
    stitch_refusal,
    stitch_operator_kwargs,
)


# ---------------------------------------------------------------------------------------
# scope: ONE control surface, and it is NOT on the operator panel (Defect 2)
# ---------------------------------------------------------------------------------------
#
# This block used to test `scope_options`, a per-panel scope combo. It is deleted, and so is
# the function. Scope belongs to the RUN, not to the operator: `_explore.resolve_run_scope`
# is the single owner and pane 1's "run on" selector is its control. The panel combo was
# wrong in both of its states -- always stale (built once, from an empty selection) and, in
# its only reachable state, mislabeled (it said "Whole dataset" while sending regions=None,
# which run_operator hands to the run selector anyway).
#
# What replaces the coverage: tests/test_explore.py's resolve_run_scope and
# describe_run_target tests, plus test_the_panel_does_not_carry_its_own_scope below.

# ---------------------------------------------------------------------------------------
# the stitcher's control surface -> stitch_region's kwargs
# ---------------------------------------------------------------------------------------

def test_defaults_reproduce_the_pipeline_exactly():
    """An untouched panel must launch byte-for-byte what stitch_region does unaided --
    otherwise the panel silently becomes a second set of defaults."""
    from squidmip._stitch import _ABS_THRESH, _BLEND_PX, _REL_THRESH

    kw = stitch_operator_kwargs(**STITCH_DEFAULTS)
    assert kw["blend_px"] == _BLEND_PX
    assert kw["rel_thresh"] == pytest.approx(_REL_THRESH)
    assert kw["abs_thresh"] == pytest.approx(_ABS_THRESH)
    assert kw["register"] is True
    assert kw["channels"] is None                 # all channels
    assert kw["registration_channel"] is None     # = the first, stitch_region's own rule


def test_the_outlier_percentage_becomes_a_fraction():
    """maragall/stitcher shows 'Outlier rel: 50%'; two_round_optimization wants 0.5. The
    conversion happens ONCE, here -- a panel that handed 50 straight through would reject
    nothing and the control would look like it worked."""
    kw = stitch_operator_kwargs(**{**STITCH_DEFAULTS, "outlier_rel_pct": 25})
    assert kw["rel_thresh"] == pytest.approx(0.25)


def test_registration_off_drops_the_registration_only_knobs():
    """With register=False there is no pose graph, so a blunder threshold is meaningless.
    Passing one anyway would let the panel show a knob that provably does nothing."""
    kw = stitch_operator_kwargs(**{**STITCH_DEFAULTS, "register": False})
    assert kw["register"] is False
    assert "rel_thresh" not in kw and "abs_thresh" not in kw
    assert "registration_channel" not in kw
    assert kw["blend_px"] == STITCH_DEFAULTS["blend_px"]   # fusion still feathers


def test_a_channel_subset_is_passed_through_as_indices():
    kw = stitch_operator_kwargs(**{**STITCH_DEFAULTS, "channels": [0, 2]})
    assert kw["channels"] == [0, 2]


def test_selecting_every_channel_is_spelled_None_not_a_full_list():
    """stitch_region documents None as 'all'. An explicit full list is equivalent today but
    it is a second spelling of the same intent, and the memory note in that docstring is
    written against None."""
    kw = stitch_operator_kwargs(**{**STITCH_DEFAULTS, "channels": [0, 1, 2]}, n_channels=3)
    assert kw["channels"] is None


def test_an_empty_channel_selection_is_refused_rather_than_fusing_nothing():
    with pytest.raises(ValueError, match="at least one channel"):
        stitch_operator_kwargs(**{**STITCH_DEFAULTS, "channels": []})


def test_a_blend_wider_than_the_tile_is_refused():
    """A feather ramp wider than the overlap never reaches full weight and DIMS the seam --
    a subtly wrong mosaic, which is worse than a refusal."""
    with pytest.raises(ValueError, match="blend"):
        stitch_operator_kwargs(**{**STITCH_DEFAULTS, "blend_px": 4096}, tile_px=2084)


def test_the_kwargs_are_accepted_by_stitch_region_itself():
    """The load-bearing one: every key this panel emits must be a real parameter of
    stitch_region. A typo'd key would raise TypeError deep inside a worker thread, where the
    only symptom is a status line that stops updating."""
    import inspect

    from squidmip._stitch import stitch_region

    accepted = set(inspect.signature(stitch_region).parameters)
    for case in ({}, {"register": False}, {"channels": [0]}):
        kw = stitch_operator_kwargs(**{**STITCH_DEFAULTS, **case})
        assert set(kw) <= accepted, f"not parameters of stitch_region: {set(kw) - accepted}"


# ---------------------------------------------------------------------------------------
# the stitch guard, surfaced BEFORE the run
# ---------------------------------------------------------------------------------------

def test_a_labels_projector_is_refused_with_a_sentence_naming_the_way_out():
    """stitch_region raises ValueError for a labels operator: feathering blends overlapping
    tiles by a weighted average, and the mean of two object ids is a third object that does
    not exist. Discovering that at the end of a multi-minute run is a bad way to learn it,
    so the panel asks the same registry first."""
    why = stitch_refusal("cellpose")
    assert why is not None
    assert "cellpose" in why
    assert "label" in why.lower()
    assert "per FOV" in why or "intensity" in why    # it must say what to do instead


def test_a_plane_op_is_no_longer_refused():
    """THE REGRESSION THIS GUARD ONCE WAS. Refusing a plane-op was right while the pipeline
    fused with z pinned to 1; IMA-277's per-plane fusion removed that refusal from
    stitch_region, and this pre-check went on blocking it in the GUI alone. A pre-check that
    outlives the guard it mirrors is worse than no pre-check: it is the engine's answer,
    wrong, delivered with authority."""
    assert stitch_refusal("decon") is None
    assert stitch_refusal("bgsub") is None
    assert stitch_refusal("flatfield") is None


def test_a_z_reducer_is_not_refused():
    assert stitch_refusal("mip") is None
    assert stitch_refusal("decon3d") is None


def test_an_unknown_projector_is_named_rather_than_crashing_the_panel():
    why = stitch_refusal("does_not_exist")
    assert why is not None and "does_not_exist" in why


# ---------------------------------------------------------------------------------------
# the deconvolution QC verdict (the "add one more iteration?" decision)
# ---------------------------------------------------------------------------------------

def test_the_first_iteration_has_nothing_to_compare_against():
    from squidmip._decon_qc import halo_verdict

    kind, msg = halo_verdict([(2, 0.40)])
    assert kind == "first"
    assert "0.40" in msg or "0.4" in msg


def test_a_falling_ratio_says_the_halo_is_still_tightening():
    from squidmip._decon_qc import halo_verdict

    kind, msg = halo_verdict([(2, 0.40), (3, 0.31)])
    assert kind == "improving"
    assert "another" in msg.lower() or "more" in msg.lower()


def test_a_rising_ratio_says_the_disc_is_growing_back():
    """The semi-convergence tell. This is the whole reason the loop is one iteration at a
    time, so it must be stated as 'stop / go back', not as a neutral number."""
    from squidmip._decon_qc import halo_verdict

    kind, msg = halo_verdict([(2, 0.31), (3, 0.44)])
    assert kind == "worse"
    assert "2" in msg                       # names the iteration to go back to


def test_the_verdict_uses_the_best_seen_not_merely_the_previous_one():
    """Falling, falling, rising, rising: the answer is still 'the best was k=3', not
    'k=4 was better than k=5'."""
    from squidmip._decon_qc import halo_verdict

    kind, msg = halo_verdict([(1, 0.9), (2, 0.5), (3, 0.3), (4, 0.4), (5, 0.6)])
    assert kind == "worse"
    assert "3" in msg


def test_an_empty_history_is_refused_rather_than_returning_a_confident_nothing():
    from squidmip._decon_qc import halo_verdict

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

from squidmip._op_panels import DeconQCPanel, DeconQCResultView, StitcherPanel  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class _Host:
    """The slice of PlateWindow the panels actually use. Small on purpose: if a panel needs
    more than this, that is a coupling worth seeing in the diff."""

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

    def explore_scopes(self):
        return []

    def say(self, text):
        self.said.append(text)

    def publish_qc_result(self, view, title):
        self.published.append((view, title))


def test_the_stitcher_panel_builds_and_offers_the_ported_controls(qapp):
    p = StitcherPanel(_Host())
    # The Settings group of maragall/stitcher, minus the parts squidmip pins (see the
    # module docstring for what was deliberately not ported and why).
    assert p.register_cb is not None
    assert p.reg_channel_combo.count() == 2
    assert p.blend_spin.value() == STITCH_DEFAULTS["blend_px"]
    assert p.rel_spin.value() == STITCH_DEFAULTS["outlier_rel_pct"]
    assert p.abs_spin.value() == STITCH_DEFAULTS["outlier_abs_px"]


def test_the_panel_does_not_carry_its_own_scope(qapp):
    """Defect 2: scope belongs to the RUN. One representation, owned by pane 1's selector."""
    p = StitcherPanel(_Host())
    assert not hasattr(p, "scope_combo")


def test_the_run_leaves_scope_unresolved_so_the_run_selector_owns_it(qapp):
    """regions=None is UNSCOPED, not 'the whole plate'. run_operator resolves it against the
    LIVE selection -- which is the whole point: the panel is built once and cached, so any
    region list it captured would be stale by the time the user pressed Run."""
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


def test_a_labels_projector_disables_the_run_button_and_says_why(qapp):
    host = _Host()
    p = StitcherPanel(host)
    p.projector_combo.setCurrentText("cellpose")
    assert not p.run_btn.isEnabled()
    assert host.said and "label" in host.said[-1].lower()
    p.projector_combo.setCurrentText("mip")
    assert p.run_btn.isEnabled()


def test_a_plane_op_projector_leaves_the_run_button_enabled(qapp):
    """The button must follow the ENGINE, not a guard the engine outgrew. Per-plane fusion
    (IMA-277) made a plane-op stitchable; a disabled button would be the GUI refusing on its
    own authority something stitch_region performs."""
    host = _Host()
    p = StitcherPanel(host)
    p.projector_combo.setCurrentText("decon")
    assert p.run_btn.isEnabled()


def test_the_run_handler_itself_refuses_labels_not_just_the_disabled_button(qapp):
    """Two defences, and this test must exercise the SECOND one.

    An earlier version clicked the button and passed because the button was disabled --
    it never entered the handler at all, so deleting the guard inside `_run` left it green.
    The guard matters independently: the button's enabled state is driven by a combo signal,
    and anything that invokes the run without going through that signal (a shortcut, a
    programmatic call, a future 'run all operators' path) must still be refused.
    """
    host = _Host()
    p = StitcherPanel(host)
    p.projector_combo.setCurrentText("cellpose")
    p.run_btn.setEnabled(True)                  # simulate reaching _run some other way
    p._run()
    assert host.calls == [], "the run must not start"
    assert "label" in host.said[-1].lower()


def test_the_decon_panel_starts_at_the_qc_start_iteration_count(qapp):
    from squidmip._decon import QC_START_ITERATIONS

    p = DeconQCPanel(_Host())
    assert p.iter_spin.value() == QC_START_ITERATIONS


def test_the_decon_panel_add_one_button_advances_by_exactly_one(qapp):
    p = DeconQCPanel(_Host())
    p.iter_spin.setValue(2)
    p.plus_btn.click()
    assert p.iter_spin.value() == 3


def test_the_decon_panel_shutdown_joins_a_running_worker(qapp):
    """Closing the Decon QC tab mid-run must JOIN the RL thread, not drop the last reference to a
    running QThread (which aborts the interpreter). _dispose_tab_widget calls shutdown(), so the
    panel must expose exactly that name — stop() alone was never on the teardown path."""
    import threading

    from squidmip._op_panels import _DeconQCWorker

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
    """Pane 3 shows the picture squidmip._decon_qc produced -- it does not build one."""
    pytest.importorskip("matplotlib")
    from squidmip._decon_qc import qc_composite

    volume = np.zeros((5, 40, 40), dtype=np.float32)
    volume[2, 20, 20] = 1000.0
    composite = qc_composite(volume, (2, 20, 20), gap=2)
    view = DeconQCResultView("A1/0/c0")
    view.show_iteration(3, composite, 0.31, "improving", "still tightening")
    img = view.image_label.pixmap().toImage()
    assert (img.width(), img.height()) == (composite.shape[1], composite.shape[0])
    assert "3" in view.caption_label.text()


def _click(qapp, view, row, col):
    """A REAL mouse press on the picture at composite pixel (row, col).

    Goes through `_ClickableImage.mousePressEvent`, so the centring offset of a pixmap inside a
    wider label is exercised rather than assumed: emitting `clicked` directly would test the
    mapping and skip the half of this that has actually been wrong in Qt code before.
    """
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
    from squidmip._decon_qc import qc_composite

    view = DeconQCResultView("A1/0/c0")
    view.show_iteration(3, qc_composite(volume, centre, view_half=view_half), 0.31,
                        "improving", "still tightening",
                        volume=volume, centre=centre, view_half=view_half)
    view.resize(400, 400)
    view.show()
    qapp.processEvents()
    return view


def test_clicking_the_picture_moves_the_crosshairs_and_re_cuts_the_strips(qapp):
    """Julio: "we should be able to ... click on there image and it moves teh crosshairs to
    display XZ and YZ bands."

    The x-z and y-z strips are sections through ONE point, and the point the run picked is the
    brightest structure it found, not necessarily the one worth judging. A click re-sections the
    SAME volume: qc_composite already takes `centre`, so no RL run happens here.
    """
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
    """A gap pixel points at no section. Snapping to the nearest one would move the crosshairs
    somewhere the user did not click, which reads as the feature working."""
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
    """The three-argument show_iteration still works and must not raise on a click: there is
    nothing to re-slice, so the picture just sits there."""
    pytest.importorskip("matplotlib")
    from squidmip._decon_qc import qc_composite

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
    """The seam that makes the click possible at all: the RL volume has to survive the worker.

    It used to emit the composite alone, which is a picture — you cannot cut a different
    section out of a picture. Pinned as a shape, not a run: `_on_done` reads `frame.volume` /
    `frame.centre` / `frame.view_half` and passes them to the view.
    """
    pytest.importorskip("matplotlib")
    from squidmip._decon_qc import qc_composite
    from squidmip._op_panels import QCFrame

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
    from squidmip._decon_qc import qc_composite

    volume = np.zeros((5, 20, 20), dtype=np.float32)
    volume[2, 10, 10] = 1000.0
    c = qc_composite(volume, (2, 10, 10), gap=2)
    view = DeconQCResultView("A1/0/c0")
    view.show_iteration(2, c, 0.40, "first", "")
    view.show_iteration(3, c, 0.31, "improving", "")
    assert [k for k, _ in view.history] == [2, 3]


# ---------------------------------------------------------------------------------------
# Defect 1: the controls ported from maragall/stitcher, and their kwargs
# ---------------------------------------------------------------------------------------

def test_every_kwarg_the_panel_emits_is_a_real_stitch_region_parameter():
    """The existing guard, re-run over the NEW keys. A typo'd key raises TypeError inside a
    worker thread, where the only symptom is a status line that stops updating."""
    import inspect

    from squidmip._stitch import stitch_region

    kw = stitch_operator_kwargs(
        register=True, registration_channel=0, channels=None, blend_px=64,
        outlier_rel_pct=50, outlier_abs_px=2, correct_distortion=True, registration_t=3)
    allowed = set(inspect.signature(stitch_region).parameters)
    assert set(kw) <= allowed, set(kw) - allowed


def test_auto_blend_is_spelled_None_all_the_way_down():
    """stitch_region measures the overlap when blend_px is None. Sending the spin's stale
    number instead would look identical in the UI and silently ignore the checkbox."""
    kw = stitch_operator_kwargs(
        register=True, registration_channel=0, channels=None, blend_px=999,
        outlier_rel_pct=50, outlier_abs_px=2, auto_blend=True)
    assert kw["blend_px"] is None


def test_auto_blend_skips_the_ramp_vs_tile_refusal():
    """The 'ramp must fit inside the tile' check is about a number the USER typed. With Auto
    on there is no such number yet -- refusing here would block the control that exists to
    compute a safe one."""
    kw = stitch_operator_kwargs(
        register=True, registration_channel=0, channels=None, blend_px=5000,
        outlier_rel_pct=50, outlier_abs_px=2, auto_blend=True, tile_px=2084)
    assert kw["blend_px"] is None


def test_distortion_and_timepoint_are_dropped_when_registration_is_off():
    """Both are registration-only. Forwarding correct_distortion=True with register=False
    would make stitch_region refuse a run the user could not see they had configured."""
    kw = stitch_operator_kwargs(
        register=False, registration_channel=None, channels=None, blend_px=64,
        outlier_rel_pct=50, outlier_abs_px=2, correct_distortion=True, registration_t=2)
    assert "correct_distortion" not in kw
    assert "registration_t" not in kw


def test_the_panel_offers_the_distortion_and_auto_blend_controls(qapp):
    p = StitcherPanel(_Host())
    assert p.distortion_cb is not None
    assert p.blend_auto_cb is not None
    # ON by default, changed deliberately on 2026-08-03 (Julio: "Correct lens distort should be
    # defaulted to on"). It used to assert the opposite. The port carried the standalone's
    # opt-in SPELLING across without its behaviour: maragall/stitcher's checkbox is dead, so the
    # standalone corrects distortion on every run, and squidmip was the only one of the two that
    # did not unless asked.
    assert p.distortion_cb.isChecked()


def test_auto_blend_disables_the_manual_width_so_no_dead_number_is_shown(qapp):
    p = StitcherPanel(_Host())
    p.blend_auto_cb.setChecked(True)
    assert not p.blend_spin.isEnabled()
    p.blend_auto_cb.setChecked(False)
    assert p.blend_spin.isEnabled()


def test_the_distortion_checkbox_is_greyed_out_with_registration_off(qapp):
    """maragall/stitcher's own version of this checkbox is never read at all (app.py:1472).
    Ours must at least not look adjustable when it provably does nothing."""
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


def test_the_stitch_defaults_are_read_off_stitch_region_not_mirrored_by_hand():
    """The dict used to import three PRIVATE constants out of `_stitch` (`_ABS_THRESH`,
    `_BLEND_PX`, `_REL_THRESH`) and re-state their values here, which is a hand-kept second copy
    of the pipeline's numbers: rename one and the panel silently launches a different run from
    the one it claims to reproduce. They come off `stitch_region`'s own signature now.

    NOT off a declaration, and the reason is worth pinning: `stitch` is a REGION operator, and
    `add_region_operator` carries no `params=` at all -- so unlike a projector there is no `Param`
    record to read. If the region table ever grows one, `_stitch_default` is the single place that
    changes.
    """
    from inspect import signature

    from squidmip._op_panels import _stitch_default
    from squidmip._stitch import stitch_region

    sig = signature(stitch_region).parameters
    assert STITCH_DEFAULTS["register"] == sig["register"].default
    assert STITCH_DEFAULTS["outlier_rel_pct"] == int(round(sig["rel_thresh"].default * 100))
    assert STITCH_DEFAULTS["outlier_abs_px"] == int(round(sig["abs_thresh"].default))
    assert STITCH_DEFAULTS["registration_t"] == sig["registration_t"].default
    assert _stitch_default("register") is sig["register"].default


# =======================================================================================
# 3. THE GENERIC PANEL: a declared Param becomes a widget
# =======================================================================================
#
# The gap this closes, stated once: `_engine.Operator` has declared `params` since Cellpose became
# a real operator, four things read that declaration (bind, the CLI's --param, _recipe, _compose)
# and the GUI was not one of them. `spot` and `cellpose` declare four parameters each and NOT ONE
# was reachable from a panel -- so an operator declaring `params=(Param("sigma", 2.0),)` got zero
# widgets and ran silently at its defaults, while `templates/operator/README.md` told contributors
# to declare them. These tests are over `squidmip._param_panel`, whose policy half is Qt-free for
# the same reason the stitcher's is.

from squidmip._engine import Param  # noqa: E402
from squidmip._param_panel import (  # noqa: E402
    WIDGET_KINDS,
    GenericOperatorPanel,
    group_params,
    panel_refusal,
    param_step,
    unsupported_params,
    widget_kind,
)


# -- the mapping rule -------------------------------------------------------------------

def test_the_widget_is_chosen_from_the_type_of_the_default():
    """THE mapping rule. `Param` declares no type, no range and no widget hint (its docstring
    forbids one), so the default's type is the only thing there is to dispatch on."""
    assert widget_kind(2.0) == "decimal"
    assert widget_kind(30) == "spin"
    assert widget_kind(True) == "check"
    assert widget_kind("nuclei") == "text"


def test_a_bool_is_a_check_box_and_not_a_zero_one_spinner():
    """`bool` is a subclass of `int`, so `isinstance(True, int)` is True. An isinstance ladder in
    the wrong order renders every check box as a 0/1 spin -- which is why the table is looked up
    by EXACT type. This is the test that would catch the ladder being written."""
    assert widget_kind(True) == "check"
    assert widget_kind(False) == "check"
    assert WIDGET_KINDS[bool] != WIDGET_KINDS[int]


def test_a_default_this_panel_cannot_draw_is_named_rather_than_guessed():
    """No silent fallback to a text box. A guessed widget is how a value the user typed becomes a
    value the run did not receive, which is the whole defect this module exists to end."""
    assert widget_kind(None) is None
    assert widget_kind((1, 2)) is None
    bad = unsupported_params((Param("sigma", 2.0), Param("mask", None)))
    assert bad == [("mask", "NoneType")]


# -- chains -----------------------------------------------------------------------------

def test_a_chain_s_namespaced_parameters_are_split_the_way_compose_joins_them():
    """`_compose` namespaces `<step>.<param>` and `_rebinder` splits on `partition(".")`. Splitting
    differently here would route a value to a step that never asked for it."""
    assert param_step("spot.min_area_px") == ("spot", "min_area_px")
    assert param_step("min_area_px") == (None, "min_area_px")


def test_a_chain_is_grouped_by_step_in_chain_order_not_refused():
    """Task 4: what does `projector_params()` return for a chain, and does the panel handle it?
    It returns the parts' params namespaced, and this is the handling: one group per step, in the
    order the expression is written."""
    from squidmip._engine import projector_params

    params = projector_params("bgsub + spot")
    assert [p.name for p in params] == ["spot.sigma_px", "spot.min_area_px",
                                        "spot.min_distance_px", "spot.split_touching"]
    groups = group_params(params)
    assert [step for step, _ in groups] == ["spot"]
    assert [p.name for p in groups[0][1]] == [p.name for p in params]


def test_a_bare_operator_s_parameters_are_one_unnamed_group():
    groups = group_params((Param("sigma_px", 2.0), Param("min_area_px", 30)))
    assert [step for step, _ in groups] == [None]


# -- the refusal ------------------------------------------------------------------------

def test_a_parameterised_operator_is_not_refused():
    assert panel_refusal("spot") is None
    assert panel_refusal("cellpose") is None


def test_a_region_operator_is_refused_because_its_table_declares_no_params():
    """`add_region_operator` carries one callable and a `requires` tuple -- no `params=` at all.
    That asymmetry between the two tables is exactly why `STITCH_DEFAULTS` still exists, so the
    refusal says it rather than reporting `stitch` as an unknown projector."""
    why = panel_refusal("stitch")
    assert why and "REGION operator" in why and "StitcherPanel" in why


def test_a_key_that_is_not_an_operator_is_refused_by_name():
    """`minerva` is a card, not an operator. Refused with the registry's own sentence."""
    why = panel_refusal("minerva")
    assert why and "minerva" in why


def test_an_undrawable_parameter_refuses_the_whole_panel_naming_the_parameter():
    """A panel that silently omitted the one parameter it could not draw would run that parameter
    at its default while every other control implied the form was complete."""
    from squidmip import add_projector

    def _factory(**kwargs):
        def _op(planes):
            return next(iter(planes))
        return _op

    name = "panel_test_undrawable"
    add_projector(name, _factory, params=(Param("sigma_px", 2.0), Param("mask", None)))
    try:
        why = panel_refusal(name)
        assert why and "mask" in why and "NoneType" in why
    finally:
        from squidmip._engine import _PROJECTORS
        _PROJECTORS.pop(name, None)


# -- the Qt half ------------------------------------------------------------------------

def test_the_panel_builds_one_widget_per_declared_parameter(qapp):
    """The defect itself, pinned: `spot` declares four parameters and had zero widgets."""
    from squidmip._engine import projector_params

    p = GenericOperatorPanel(_Host(), "spot")
    assert sorted(p.widgets) == sorted(param.name for param in projector_params("spot"))
    assert len(p.widgets) == 4


def test_each_widget_starts_at_the_declared_default(qapp):
    """An untouched panel must launch what the operator ships with, or the panel has become a
    second set of defaults -- the same rule `STITCH_DEFAULTS` is held to."""
    from squidmip._engine import projector_params

    p = GenericOperatorPanel(_Host(), "spot")
    declared = {param.name: param.default for param in projector_params("spot")}
    assert p.kwargs() == declared


def test_the_blurb_becomes_the_tooltip(qapp):
    """`Param.blurb` is documented as 'the one line a UI shows'. Until now no UI showed it."""
    from squidmip._engine import projector_params

    blurbs = {param.name: param.blurb for param in projector_params("spot")}
    p = GenericOperatorPanel(_Host(), "spot")
    for name, widget in p.widgets.items():
        if blurbs[name]:
            assert widget.toolTip() == blurbs[name]


def test_a_value_read_back_keeps_the_declared_type(qapp):
    """`spots_op` builds a `SpotParams` dataclass out of these. A `min_area_px` arriving as 30.0
    where 30 was declared survives all the way to a comparison against an integer pixel count."""
    p = GenericOperatorPanel(_Host(), "spot")
    p.widgets["min_area_px"].setValue(400)
    kwargs = p.kwargs()
    assert kwargs["min_area_px"] == 400 and isinstance(kwargs["min_area_px"], int)
    assert isinstance(kwargs["sigma_px"], float)
    assert isinstance(kwargs["split_touching"], bool)


def test_the_widget_s_value_travels_to_run_operator_through_operator_kwargs(qapp):
    """The SAME argument StitcherPanel uses, so the value's journey to the pixels is a path that
    was already tested rather than a second one."""
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
    """The panel's output has to survive `Operator.bind`, which refuses an unknown name LOUD.
    A panel emitting a key the operator does not declare would raise inside a worker thread,
    where the only symptom is a status line that stops updating."""
    from squidmip import bind_projector

    host = _Host()
    p = GenericOperatorPanel(host, "spot")
    p.widgets["min_area_px"].setValue(80)
    p.run_btn.click()
    bind_projector("spot", host.calls[0][1]["operator_kwargs"])   # raises if a name is wrong


def test_a_plane_op_is_offered_preview_only_and_the_choice_comes_off_consumes(qapp):
    """`spot` keeps z at full depth, so there is no plate to save. Read off `consumes`, never off
    the name -- test_operator_declaration fails the build on a name comparison."""
    p = GenericOperatorPanel(_Host(), "spot")
    assert p._reduces_z is False
    # Not built at all rather than built and left out of the layout: `_viewer._raw_btn` is the
    # precedent for an orphan QPushButton popping up as its own floating window.
    assert p.save_btn is None


def test_a_z_reducer_is_offered_the_save_run(qapp):
    p = GenericOperatorPanel(_Host(), "mip")
    assert p._reduces_z is True
    assert p.save_btn is not None and p.save_btn.parent() is not None


def test_an_operator_with_no_parameters_still_builds_and_says_so(qapp):
    """`mip` declares nothing. A panel that refused would make 'no parameters' and 'unknown
    operator' look identical, which is the rule `available_projectors` is written to."""
    p = GenericOperatorPanel(_Host(), "mip")
    assert p.widgets == {}
    assert p.kwargs() == {}


def test_a_chain_panel_shows_the_form_and_greys_the_run_with_a_reason(qapp):
    """Buildable and launchable are two questions. A chain's params are readable; `run_operator`
    gates on `runnable_operators()`, which lists table keys and has never held an expression. So
    the form is shown and the buttons say why they are off -- not a click that dies elsewhere."""
    host = _Host()
    p = GenericOperatorPanel(host, "bgsub + spot")
    assert sorted(p.widgets) == ["spot.min_area_px", "spot.min_distance_px",
                                 "spot.sigma_px", "spot.split_touching"]
    assert not p.run_btn.isEnabled()
    assert host.said and "chain" in host.said[-1]


def test_a_chain_panel_keeps_the_namespaced_names_bind_expects(qapp):
    """`Operator.bind` validates against the namespaced tuple this panel was built from, so the
    values go back under exactly the names they arrived with."""
    from squidmip import bind_projector

    p = GenericOperatorPanel(_Host(), "bgsub + spot")
    p.widgets["spot.min_area_px"].setValue(400)
    kwargs = p.kwargs()
    assert kwargs["spot.min_area_px"] == 400
    bind_projector("bgsub + spot", kwargs)        # raises if a namespaced name is wrong
