"""The ONE parameter surface: panels generated from the declaration, inline.

The bespoke operator PAGES and the decon QC sweep are shelved whole (Julio, 2026-08-25:
"You should shelf those operator pages"; "The sweep code should be shelved. I can just run
on an ROI iteration by iteration."). Absences are pinned here; the surviving surface is
GenericOperatorPanel (headline + the "advanced parameters" disclosure), the z-handling
combo's keep-every-plane vocabulary, RegisterPanel's copy switch and DeconPanel's NI row.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

from qtpy.QtWidgets import QApplication, QComboBox  # noqa: E402

import squidxplorer._op_panels as OP  # noqa: E402
from squidxplorer._op_panels import KEEP_EVERY_PLANE, z_operator_choice  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class _Host:
    """The minimal host surface a panel touches."""

    def __init__(self):
        self.said = []
        self._meta = {"channels": [{"name": "405"}, {"name": "488"}],
                      "frame_shape": (64, 64), "n_z": 3}
        self._reader = object()

    def say(self, text):
        self.said.append(text)


# --- the shelf: pages and the sweep are GONE WHOLE ------------------------------------------


def test_the_operator_pages_and_the_sweep_are_gone_whole():
    for name in ("StitcherPanel", "DeconQCPanel", "DeconQCResultView", "_DeconQCWorker",
                 "QCFrame", "stitch_operator_kwargs", "stitch_refusal", "_stitch_declared",
                 "_shipped_iterations"):
        assert not hasattr(OP, name), f"_op_panels still ships {name}"


def test_the_sweep_module_is_gone_whole():
    import importlib.util

    assert importlib.util.find_spec("squidxplorer._decon_qc") is None, (
        "_decon_qc (the QC sweep) is back; it was shelved whole")


def test_the_snapshot_capture_hook_is_gone():
    import inspect

    from squidxplorer import _decon, _decon_gpu

    assert "snapshot_iters" not in inspect.signature(_decon._run).parameters
    assert "snapshot_iters" not in inspect.signature(_decon_gpu.rl).parameters


def test_the_plate_publishes_no_qc_tab():
    import squidxplorer._viewer as V

    assert not hasattr(V.PlateWindow, "publish_qc_result"), (
        "publish_qc_result is back; results land as data layers in the asking view")


# --- the keep-every-plane vocabulary ---------------------------------------------------------


def test_the_keep_every_plane_label_spells_none():
    assert z_operator_choice(KEEP_EVERY_PLANE) is None
    assert z_operator_choice("mip") == "mip"


# --- the generated panels --------------------------------------------------------------------


def test_stitchs_panel_is_generated_and_its_z_handling_is_a_combo(qapp):
    from squidxplorer._param_panel import GenericOperatorPanel

    panel = GenericOperatorPanel(_Host(), "stitch")
    widget = panel.widgets["z_operator"]
    assert isinstance(widget, QComboBox), "the inner operator must be a choice, not free text"
    labels = [widget.itemText(i) for i in range(widget.count())]
    assert KEEP_EVERY_PLANE in labels, "keep-every-plane fell out of the z-handling combo"
    i = widget.findText(KEEP_EVERY_PLANE)
    widget.setCurrentIndex(i)
    assert panel.kwargs()["z_operator"] is None, (
        "the keep-every-plane label must reach the run as z_operator=None")


def test_the_advanced_disclosure_is_named_advanced_parameters(qapp):
    from squidxplorer._param_panel import GenericOperatorPanel

    panel = GenericOperatorPanel(_Host(), "stitch")
    assert panel.adv_btn is not None, "stitch declares advanced params; the disclosure is gone"
    assert panel.adv_btn.text() == "advanced parameters", (
        "Julio, 2026-08-25: the disclosure's label is exactly 'advanced parameters'")
    assert not panel._advanced.isVisibleTo(panel), "the disclosure must start collapsed"
    adv_names = {"registration_channel", "registration_t", "correct_illumination"}
    assert adv_names <= set(panel.widgets), "the advanced knobs are not in the panel"


def test_set_param_writes_the_inner_combo_and_none_maps_to_the_label(qapp):
    from squidxplorer._param_panel import GenericOperatorPanel

    panel = GenericOperatorPanel(_Host(), "stitch")
    assert panel.set_param("z_operator", None) is None
    assert panel.widgets["z_operator"].currentText() == KEEP_EVERY_PLANE
    assert panel.set_param("z_operator", "mip") is None
    assert panel.kwargs()["z_operator"] == "mip"


def test_registers_panel_keeps_the_copy_switch(qapp):
    from squidxplorer._param_panel import RegisterPanel

    panel = RegisterPanel(_Host())
    assert panel.copy_check.isChecked(), "the copy is the operator's purpose; default ON"
    assert panel.kwargs().get("copy") is True
    panel.copy_check.setChecked(False)
    assert "copy" not in panel.kwargs()


def test_decons_panel_is_iterations_plus_ni(qapp):
    from squidxplorer import _decon
    from squidxplorer._param_panel import DeconPanel

    before = _decon.session_ni()
    try:
        panel = DeconPanel(_Host())
        assert "iterations" in panel.widgets, "iterations is decon's headline knob"
        panel.ni_spin.setValue(1.33)
        assert _decon.session_ni() == pytest.approx(1.33), (
            "the NI row must write the session index the PSF reads")
    finally:
        _decon.set_session_ni(before)


def test_a_param_tooltip_is_at_most_one_sentence(qapp):
    from squidxplorer._param_panel import GenericOperatorPanel, _one_sentence

    assert _one_sentence("First. Second. Third.") == "First."
    assert _one_sentence("counts added to every plane") == "counts added to every plane", \
        "a tooltip is the declaration's own words: no invented trailing period"
    assert _one_sentence("") == ""
    panel = GenericOperatorPanel(_Host(), "stitch")
    for name, widget in panel.widgets.items():
        tip = widget.toolTip()
        assert tip.count(". ") == 0, (
            f"{name}'s tooltip carries more than one sentence: {tip!r}")
