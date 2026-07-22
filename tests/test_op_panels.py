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
    plane_op_refusal,
    scope_options,
    stitch_operator_kwargs,
)


# ---------------------------------------------------------------------------------------
# scope: ONE control surface, in pane 1 (the selector replaces a second set of buttons)
# ---------------------------------------------------------------------------------------
#
# A UI audit found two operator registries launching the same operators from pane 1 and
# pane 3 with different labels and different `save` defaults, and they had diverged in
# production. The fix is not a third caller: it is that "run this on the subset parked in
# pane 3" is a SCOPE VALUE on the pane-1 panel, not a button in pane 3.

def test_whole_dataset_is_always_offered_and_is_the_first_choice():
    opts = scope_options(["A1", "A2", "A3"], selected=[], explore_scopes=[])
    assert len(opts) == 1
    label, regions = opts[0]
    assert regions is None                     # None is run_operator's "whole plate"
    assert "3" in label                        # the count is shown, not just the word


def test_a_plate_selection_becomes_a_scope_choice():
    opts = scope_options(["A1", "A2", "A3"], selected=["A2", "A3"], explore_scopes=[])
    assert [r for _, r in opts] == [None, ["A2", "A3"]]
    assert "2" in opts[1][0]


def test_each_pane3_subset_is_a_scope_choice_not_a_button_in_pane_3():
    opts = scope_options(["A1", "A2", "A3"], selected=[],
                         explore_scopes=[("A1+A2", ["A1", "A2"]), ("A3", ["A3"])])
    assert [r for _, r in opts] == [None, ["A1", "A2"], ["A3"]]
    assert "A1+A2" in opts[1][0]


def test_an_empty_subset_is_not_offered_as_a_runnable_scope():
    """Choosing it would produce 'empty selection -- nothing to run' at the far end of the
    click. Refusing to OFFER it is the honest version."""
    opts = scope_options(["A1"], selected=[], explore_scopes=[("stale", [])])
    assert [r for _, r in opts] == [None]


def test_scope_regions_are_copies_so_a_later_plate_click_cannot_rewrite_a_pending_run():
    selected = ["A1"]
    opts = scope_options(["A1", "A2"], selected=selected, explore_scopes=[])
    selected.append("A2")
    assert opts[1][1] == ["A1"]


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
# the plane-op guard, surfaced BEFORE the run
# ---------------------------------------------------------------------------------------

def test_a_plane_op_projector_is_refused_with_a_sentence_naming_the_way_out():
    """stitch_region raises NotImplementedError for a plane-op (IMA-277). Discovering that
    at the end of a multi-minute run is a bad way to learn it, so the panel asks first."""
    why = plane_op_refusal("decon")
    assert why is not None
    assert "decon" in why
    assert "mip" in why or "decon3d" in why      # it must say what to do instead


def test_a_z_reducer_is_not_refused():
    assert plane_op_refusal("mip") is None
    assert plane_op_refusal("decon3d") is None


def test_an_unknown_projector_is_named_rather_than_crashing_the_panel():
    why = plane_op_refusal("does_not_exist")
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
pytest.importorskip("PyQt5")

import sys  # noqa: E402

if "PySide6" in sys.modules or "PySide2" in sys.modules:   # pragma: no cover
    pytest.skip("a PySide binding is already loaded", allow_module_level=True)

from PyQt5.QtWidgets import QApplication  # noqa: E402

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
    assert p.scope_combo.count() >= 1


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


def test_a_plane_op_projector_disables_the_run_button_and_says_why(qapp):
    host = _Host()
    p = StitcherPanel(host)
    p.projector_combo.setCurrentText("decon")
    assert not p.run_btn.isEnabled()
    assert host.said and "plane-op" in host.said[-1]
    p.projector_combo.setCurrentText("mip")
    assert p.run_btn.isEnabled()


def test_the_run_handler_itself_refuses_a_plane_op_not_just_the_disabled_button(qapp):
    """Two defences, and this test must exercise the SECOND one.

    An earlier version clicked the button and passed because the button was disabled --
    it never entered the handler at all, so deleting the guard inside `_run` left it green.
    The guard matters independently: the button's enabled state is driven by a combo signal,
    and anything that invokes the run without going through that signal (a shortcut, a
    programmatic call, a future 'run all operators' path) must still be refused.
    """
    host = _Host()
    p = StitcherPanel(host)
    p.projector_combo.setCurrentText("decon")
    p.run_btn.setEnabled(True)                  # simulate reaching _run some other way
    p._run()
    assert host.calls == [], "the run must not start"
    assert "plane-op" in host.said[-1]


def test_the_decon_panel_starts_at_the_qc_start_iteration_count(qapp):
    from squidmip._decon import QC_START_ITERATIONS

    p = DeconQCPanel(_Host())
    assert p.iter_spin.value() == QC_START_ITERATIONS


def test_the_decon_panel_add_one_button_advances_by_exactly_one(qapp):
    p = DeconQCPanel(_Host())
    p.iter_spin.setValue(2)
    p.plus_btn.click()
    assert p.iter_spin.value() == 3


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
