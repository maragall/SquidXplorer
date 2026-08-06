"""The two HAND-WRITTEN operator interfaces: the stitcher's controls and decon's QC loop.

THESE TWO, AND ONLY THESE TWO (2026-08-05). Every other operator's panel is BUILT FROM ITS
``params`` DECLARATION by :mod:`squidmip._param_panel`, which asks the registry what an operator
takes and maps each :class:`~squidmip._engine.Param` to a widget by the type of its default. This
module used to be the only source of an operator panel, and the cost was measured: an operator
declaring ``params`` got zero widgets and ran silently at its defaults, so ``spot`` and
``cellpose`` declared four parameters each and not one was reachable from any panel.

The two below stay hand-written because they do things a parameter FORM cannot, and deleting them
to gain uniformity would delete real behaviour: :class:`StitcherPanel` converts a percentage to a
fraction, greys out the knobs that provably do nothing with registration off, and refuses a
labels operator with a sentence before the run starts; :class:`DeconQCPanel` runs an iterative
semi-convergence loop and publishes a picture as a tab of its own. ``_viewer._activate_operator`` prefers a
hand-written panel and falls back to the generic one, so adding an operator needs no edit here.

Julio: "Right now I'm blocked in testing the post-processing because Stitcher doesn't have
that maragall/Stitcher interface embedded in our top-left subpane. The deconvolution is not
showing the XZ/YZ strips on the turbo colormap on the exploration pane so that we can choose
the iterations."

WHERE THINGS LIVE, AND WHY
--------------------------
* Every CONTROL is on the ONE operator panel. A UI audit found two operator registries
  (``_OPERATIONS`` and ``runnable_operators()``) launching the same operators from two
  different panes with different labels and different ``save`` defaults, and a comment in
  ``_viewer.py`` records that they diverged in production. A target is a SCOPE VALUE on the
  one run selector, not a second set of buttons somewhere else. This module adds no third
  caller to either registry.
* The deconvolution RESULT - the 2-D image in turbo with the x-z and y-z strips concatenated
  to it - is a TAB (:class:`DeconQCResultView`) opened beside these controls. That is where a
  preview result is looked at, and it is big: it needs the room.

THE SEAM WITH THE WINDOW, STATED NARROWLY
-----------------------------------------
This module never touches a tab bar. It calls exactly one method on its host::

    host.publish_qc_result(widget, title)   # -> shows `widget` as a tab called `title`

``PlateWindow`` implements that with its EXISTING ``_open_op_tab``, so no new tab API is
introduced. If a host does not implement it, the panel SAYS SO in the readout; the picture is
never computed and then dropped. It used to land in the exploration pane, which between
2b8fbc5 and 2026-08-05 was not in any layout — computed, tabbed, and shown to nobody.

WHAT WAS PORTED FROM maragall/stitcher, AND WHAT WAS NOT
--------------------------------------------------------
Ported (its ``Settings`` group, ``gui/app.py``):

* "Enable registration refinement" -> ``register=``, which is also the ``stitch`` vs
  ``coordinate`` operator choice already in ``_stitch.py``.
* Registration channel -> ``registration_channel=``.
* "Blend pixels" -> ``blend_px=``.
* "Outlier rel: N%" / "abs: N px" -> ``rel_thresh=`` / ``abs_thresh=``. These were module
  constants; IMA-decon-stitch-ui threads them through ``solve_offsets_px``.
* Run button + progress + a log of per-region results.

NOT ported, each for a reason rather than for lack of time:

* **Downsample factor.** ``_DOWNSAMPLE_FACTORS = (1, 1)`` is PINNED here with a stated
  reason ("registration MUST be full-res; any downsample coarsens the sub-pixel shift").
  Exposing a knob whose whole effect is to make the answer worse, on a post-acquisition
  tool where the run is not interactive, is not a control - it is a trap.
* **Registration z-level / timepoint.** In the stitcher those pick a plane out of a stack.
  Here the projector has already reduced z before registration runs, and geometry is solved
  once on t=0 by construction. There is no plane to pick.
* **Flatfield group.** squidmip has ``_flatfield.py`` as its own operator with its own
  profile chooser. A second flat-field UI here would be a second owner of one setting.
* **"Auto" blend width** (stitcher computes ~2x the seam overlap). squidmip has no overlap
  measurement of its own - ``tilefusion.find_adjacent_pairs`` owns that geometry internally
  and does not report it back. Deriving a second estimate here would be an unvalidated
  second representation of the same number. The default is the measured ``_BLEND_PX`` and
  the tooltip states the real overlap it was sized against.
* **Lens-distortion correction.** This bullet used to read "not in the ``tilefusion`` call
  chain ``_stitch.py`` ports; a checkbox for a stage that does not run would be a lie". That
  stopped being true when the stage was wired (``_stitch.py``'s ``correct_distortion`` block):
  the checkbox is here, it decides, and as of 2026-08-03 it is ON by default, matching the
  standalone, where ``enable_distortion`` runs unconditionally because its checkbox is dead.
* **Preview grid (N x M), drag-and-drop, "Open in Napari", "Export OME-TIFF", "Open
  Existing", "Max Projection".** The stitcher is a standalone app that has to load a
  dataset and then hand its result to a viewer. This IS the viewer: the plate is already
  open, and the fused mosaic lands in the embedded napari by itself.

FROM maragall/deconvolution (``petakit``'s ``gui/main.py``)
----------------------------------------------------------
Ported: the iterations spinner with a "recommended: N" hint beside it, the "Force CPU"
checkbox, the channel selector, and a status line that shows the reason on failure.

Not ported: the **method combo** (petakit offers ``omw``/``rl``; ``_decon.py`` PINS ``rl``
because ``omw`` returns an all-zero volume on this instrument's geometry - offering it would
be offering a black image), the **output directory + OME-TIFF save** (this is a viewer, not
an exporter), the **drag-and-drop acquisition box** (the plate window owns dataset loading),
and the **"Preview (5 FOVs)" ComparisonWindow** (a second top-level window with two more
viewers in it, in an app whose whole point is bundling three panes into one).

Added, because petakit's GUI is fire-and-forget and Julio's loop is not: **"+1 iteration"**,
and the halo/core number with :func:`squidmip._decon_qc.halo_verdict`'s sentence next to the
picture, so "are the light halos handled" has a number beside the eye.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np
from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

#: The feather-ramp FALLBACK. The one number here with no signature default behind it:
#: ``stitch_region(blend_px=None)`` means "measure this acquisition's real overlap", and this is
#: what the pipeline falls back to when nothing overlaps. It is the spin's starting value, with the
#: "Auto" box beside it selecting the other mode, so the panel offers both of the pipeline's
#: modes rather than inventing a third number.
from squidmip import _qtstyle
from squidmip._stitch import _BLEND_PX


def _stitch_default(name: str):
    """One default, read off ``stitch_region``'s OWN SIGNATURE rather than mirrored here.

    Why this is not simply ``from squidmip._stitch import _REL_THRESH``, which is what it was:
    that spelling made the panel a SECOND copy of the pipeline's numbers, kept in step by hand and
    reaching through the module's privacy to do it. ``_ABS_THRESH`` and ``_REL_THRESH`` are not the
    contract -- they are how the contract's defaults happen to be spelled inside ``_stitch``, and
    renaming one silently left the panel launching a different run from the pipeline it claimed to
    reproduce.

    Why not from a DECLARATION, which is where a projector's parameters come from
    (``squidmip._param_panel``): ``stitch`` is a REGION operator, and ``add_region_operator``
    carries no ``params=`` at all -- one callable and a ``requires`` tuple, nothing more. There is
    no ``Param`` record for this panel to read. The signature is the closest thing to a
    declaration that this table has, and reading it is what removes the hand-kept copy. If the
    region table ever grows ``params=``, this function is the one place that changes.
    """
    from inspect import signature

    from squidmip._stitch import stitch_region

    return signature(stitch_region).parameters[name].default


#: The panel's starting position. Every value here is the pipeline's own default, so an
#: untouched panel launches byte-for-byte what ``stitch_region`` does unaided - a panel with
#: opinions of its own would be a second set of defaults to keep in step.
#:
#: ``blend_px`` is the ONE deliberate divergence and it is spelled out rather than hidden:
#: ``stitch_region``'s own default is ``None``, which means "measure this acquisition's real
#: overlap", and the panel starts at the fixed fallback with an "Auto" box beside it. Both
#: numbers reach ``stitch_region`` through ``auto_blend``, so nothing here is a second default --
#: it is which of the pipeline's two modes the box starts in.
STITCH_DEFAULTS = {
    "register": _stitch_default("register"),
    "registration_channel": _stitch_default("registration_channel"),
    "channels": _stitch_default("channels"),
    "blend_px": _BLEND_PX,
    "outlier_rel_pct": int(round(_stitch_default("rel_thresh") * 100)),
    "outlier_abs_px": int(round(_stitch_default("abs_thresh"))),
    "auto_blend": False,
    # ON (Julio, 2026-08-03: "Correct lens distort should be defaulted to on"). Same value as
    # stitch_region's own resolved default, which is what keeps this dict honest: an untouched
    # panel still launches byte-for-byte what stitch_region does unaided.
    "correct_distortion": True,
    "registration_t": _stitch_default("registration_t"),
}


# ---------------------------------------------------------------------------------------
# policy (no Qt) — the decisions, separated from the pixels
# ---------------------------------------------------------------------------------------

def stitch_operator_kwargs(*, register, registration_channel, channels, blend_px,
                           outlier_rel_pct, outlier_abs_px,
                           auto_blend: bool = False,
                           correct_distortion: bool = True,
                           registration_t: int = 0,
                           n_channels: Optional[int] = None,
                           tile_px: Optional[int] = None) -> dict:
    """Turn the panel's widget values into ``stitch_region`` keyword arguments.

    Every key returned is a real parameter of :func:`squidmip._stitch.stitch_region` - a
    test asserts that against its signature, because a typo'd key raises ``TypeError``
    inside a worker thread where the only symptom is a status line that stops updating.

    The conversions that must happen exactly once, and here:

    * ``outlier_rel_pct`` is a PERCENTAGE in the UI (maragall/stitcher shows "50%") and a
      FRACTION in ``two_round_optimization``. Handing 50 straight through would reject
      nothing while the control looked like it worked.
    * With ``register=False`` there is no pose graph, so the blunder thresholds and the
      registration channel are DROPPED rather than passed and ignored.
    * "every channel" is spelled ``None``, not a full index list - that is the spelling
      ``stitch_region``'s docstring (and its memory note) is written against.
    """
    if channels is not None:
        channels = [int(c) for c in channels]
        if not channels:
            raise ValueError(
                "select at least one channel to fuse: a mosaic with no channels is not a "
                "smaller result, it is no result.")
        if n_channels is not None and len(channels) == int(n_channels):
            channels = None
    # "Auto" is spelled blend_px=None all the way down to stitch_region, which measures the
    # acquisition's real overlap (auto_blend_px). The spin's value is IGNORED, not clamped:
    # showing a number that had no effect is the accepted-and-ignored shape.
    if auto_blend:
        blend_px = None
    else:
        blend_px = int(blend_px)
    if blend_px is not None and tile_px is not None and blend_px >= int(tile_px):
        raise ValueError(
            f"blend width {blend_px} px is not smaller than the {int(tile_px)} px tile. The "
            "Hann feather has to fit INSIDE the real overlap; a ramp that never reaches full "
            "weight dims every seam, which looks like a stitching artefact rather than a "
            "setting.")

    kwargs = {"register": bool(register), "channels": channels, "blend_px": blend_px}
    if register:
        kwargs["registration_channel"] = registration_channel
        kwargs["rel_thresh"] = float(outlier_rel_pct) / 100.0
        kwargs["abs_thresh"] = float(outlier_abs_px)
        kwargs["registration_t"] = int(registration_t)
        # Distortion correction fits the residual left AFTER the global solve, so it is
        # registration-only; stitch_region refuses the combination outright. Dropping it here
        # rather than forwarding a False keeps "what the panel sends" equal to "what the run
        # does" for the disabled case too.
        kwargs["correct_distortion"] = bool(correct_distortion)
    return kwargs


def stitch_refusal(projector: str) -> Optional[str]:
    """The sentence explaining why *projector* cannot be stitched, or ``None`` if it can.

    This MIRRORS the guard ``stitch_region`` actually has, asked of the same registry before
    the run starts, because discovering it at the end of a multi-minute fuse is a bad way to
    learn it. It is a pre-check, not a second guard - the operator's own refusal stays
    exactly where it is.

    It used to refuse a PLANE-OP, and that was right while the pipeline fused with z pinned
    to 1. IMA-277's per-plane fusion removed that refusal from ``stitch_region`` (see its
    "PER-PLANE FUSION" section: the geometry is solved once and every plane is fused from the
    same origins), so a plane-op stitches and this pre-check was blocking, in the GUI only,
    something the engine had learned to do. What ``stitch_region`` still refuses is LABELS,
    and only labels: feathering blends overlapping tiles by a weighted average, and the mean
    of label 12 and label 37 is label 24, an object that does not exist.
    """
    from squidmip._stitch import LABELS, _resolve_operator

    try:
        op = _resolve_operator(projector)
    except Exception as exc:                       # unknown name -> name it, don't crash
        return (f"{projector!r} is not a projector this build knows: {exc}")
    if op.produces != LABELS:
        return None
    return (
        f"{projector!r} produces label images (integer object ids), and fusion blends "
        f"overlapping tiles by a weighted average - the mean of two label ids is a third, "
        f"nonexistent object, and per-FOV ids collide across every seam. Segment per FOV "
        f"instead, or stitch an intensity operator such as mip or decon.")


# ---------------------------------------------------------------------------------------
# shared chrome
# ---------------------------------------------------------------------------------------

_BG = "#0d1117"
_SUB = "color:#8b98ad;font-size:11px;"
_HEAD = "color:#57606a;font-size:10px;font-weight:800;letter-spacing:1.5px;padding-top:6px;"


def _apply_qss(root: QWidget) -> None:
    """Style every control in *root* the way the rest of the operator panel is styled.

    Not a second dark theme: :mod:`squidmip._qtstyle` owns these strings and every other control
    in the operator panel is drawn with them. Screenshotting the first build is what caught this --
    unstyled QPushButtons render as flat text on this background and do not read as clickable at
    all.

    Read from ``_qtstyle`` DIRECTLY, at module scope. This used to be a ``_qss()`` helper doing
    ``from squidmip._viewer import _BTN_QSS, _CHECK_QSS, _COMBO_QSS`` inside its own body, and that
    lazy import was the exact line ``_qtstyle``'s docstring quotes as the reason ``_qtstyle``
    exists: a leaf colour fact reachable only through the 4,500-line window module, deferred into a
    function to dodge a cycle. ``_viewer`` had already been reduced to seven ``_BTN_QSS =
    _qtstyle.BTN_QSS`` aliases, so the cycle was gone and the hop was pure ceremony -- and it
    carried a live footgun, because the helper took the three names in one order and returned them
    in another.
    """
    for w in root.findChildren(QPushButton):
        w.setStyleSheet(_qtstyle.BTN_QSS)
        w.setCursor(Qt.PointingHandCursor)
    for w in root.findChildren(QComboBox):
        w.setStyleSheet(_qtstyle.COMBO_QSS)
    for w in root.findChildren(QSpinBox):
        w.setStyleSheet(_qtstyle.COMBO_QSS)
    for w in root.findChildren(QCheckBox):
        w.setStyleSheet(_qtstyle.CHECK_QSS)


def _wrapped(text: str, style: str) -> QLabel:
    """A word-wrapped QLabel that actually RESERVES the height its wrapping needs.

    A plain word-wrapped QLabel reports a single-line sizeHint to the layout, so the
    paragraph paints over whatever sits under it. In the first build the deconvolution
    blurb's third line printed on top of the "WHERE TO MEASURE" header -- visible in the
    screenshot, invisible in the source.
    """
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setStyleSheet(style)
    lab.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
    lab.setMinimumHeight(lab.fontMetrics().height() * 2)
    return lab


def _head(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setStyleSheet(_HEAD)
    return lab


def _row(*widgets) -> QHBoxLayout:
    lay = QHBoxLayout()
    lay.setSpacing(6)
    for w in widgets:
        lay.addWidget(w) if isinstance(w, QWidget) else lay.addLayout(w)
    lay.addStretch(1)
    return lay


def _channel_names(host) -> list:
    meta = getattr(host, "_meta", None) or {}
    return [c["name"] for c in meta.get("channels", [])]


class _Panel(QWidget):
    """Common shell: a title, a blurb, and a status line that is never silent."""

    def __init__(self, host, title: str, blurb: str):
        super().__init__()
        self.host = host
        self.setStyleSheet(f"background:{_BG};color:#e6edf3;")

        # A SCROLL AREA, because these are tall control stacks in a narrow pane. The
        # stitcher's alone is scope + z-reduction + registration + fusion + channels + run,
        # and the window also has to hold the plate view. Without this the bottom controls are
        # simply unreachable at ordinary window heights -- and "the canvas squeezed to a
        # 140 px sliver" is the precedent for trusting a screenshot over a layout argument.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        body = QWidget()
        self.v = QVBoxLayout(body)
        self.v.setContentsMargins(16, 14, 16, 14)
        self.v.setSpacing(8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        scroll.setWidget(body)
        outer.addWidget(scroll)

        t = QLabel(title)
        t.setStyleSheet("font-size:16px;font-weight:800;")
        self.v.addWidget(t)
        self.v.addWidget(_wrapped(blurb, "color:#8b98ad;font-size:12px;"))
        self.status = _wrapped("", "color:#d29922;font-size:11px;")

    def say(self, text: str) -> None:
        """Put a SENTENCE in front of the user. Never log-and-continue.

        Routed to the window's readout as well as the panel's own line, because the panel
        may not be the visible tab when the thing worth saying happens.
        """
        self.status.setText(text)
        say = getattr(self.host, "say", None)
        if callable(say):
            say(text)


# ---------------------------------------------------------------------------------------
# 1. the stitcher
# ---------------------------------------------------------------------------------------

class StitcherPanel(_Panel):
    """maragall/stitcher's control surface over squidmip's ``stitch`` region operator."""

    def __init__(self, host):
        super().__init__(
            host, "Stitch (register + fuse)",
            "Register every FOV of a region against its neighbours and fuse one seamless "
            "mosaic. A region is a MOSAIC of FOVs, so one region produces one image.")
        names = _channel_names(host)

        # SCOPE IS NOT HERE, DELIBERATELY (Defect 2). It belongs to the RUN, not to the
        # operator, and the window's "run on" selector owns it. This panel used to carry a second
        # scope combo, and it was wrong in both of its states:
        #
        #   * it was built ONCE and cached by _open_op_tab, from a selection read at build
        #     time. The user opens the stitcher tab BEFORE picking wells, so the selection was
        #     always empty and the combo always collapsed to its single "Whole dataset" entry
        #     -- which is the whole of "Only scope available for the stitcher is the whole
        #     dataset". It was never a capability limit; it was a stale read.
        #   * and that entry sent regions=None, which run_operator treats as "unscoped" and
        #     hands to the run selector anyway. So the control the user was looking at said
        #     "Whole dataset" while the run actually went wherever the OTHER control pointed.
        #     A mislabeled control is worse than a missing one.
        #
        # Two representations of one truth, which is this project's dominant defect shape. The
        # run selector is the owner because it is the one that reads the selection LIVE.

        # -- what to reduce z with before fusing ---------------------------------------
        self.v.addWidget(_head("Z REDUCTION"))
        self.projector_combo = QComboBox()
        from squidmip import available_projectors

        for name in sorted(available_projectors()):
            self.projector_combo.addItem(name)
        self.projector_combo.setCurrentText("mip")
        self.projector_combo.setToolTip(
            "Each FOV's z-stack is reduced to one plane before registration. Only a "
            "z-REDUCER can be stitched; a plane-op is refused with a reason.")
        self.projector_combo.currentTextChanged.connect(self._check_projector)
        self.v.addLayout(_row(QLabel("Reduce z with:"), self.projector_combo))

        # -- registration --------------------------------------------------------------
        self.v.addWidget(_head("REGISTRATION"))
        self.register_cb = QCheckBox("Enable registration refinement")
        self.register_cb.setChecked(STITCH_DEFAULTS["register"])
        self.register_cb.setToolTip(
            "On: phase-correlate overlapping pairs and solve a global pose graph.\n"
            "Off: pure coordinate placement from the stage — the honest control for judging "
            "whether registration actually helped.")
        self.register_cb.toggled.connect(self._on_register_toggled)
        self.v.addWidget(self.register_cb)

        self.reg_channel_combo = QComboBox()
        for name in names:
            self.reg_channel_combo.addItem(name)
        self.reg_channel_combo.setToolTip(
            "ONE channel drives the geometry and every channel is then fused with that one "
            "solution — channels of a FOV share a sensor and must not get disagreeing "
            "placements.")
        self.v.addLayout(_row(QLabel("Registration channel:"), self.reg_channel_combo))

        # maragall/stitcher's Timepoint spin (app.py:1428). Not cosmetics: the geometry is
        # solved at ONE timepoint, and that timepoint used to be a hardcoded 0 with no way to
        # see or set it -- the same defect CLASS as the registration-channel substitution bug
        # (a solve running somewhere the user cannot name). Hidden on a single-timepoint
        # acquisition, where the only legal value is 0 and a spin would be furniture.
        n_t = int(((getattr(host, "_meta", None) or {}).get("n_t")) or 1)
        self.reg_t_spin = QSpinBox()
        self.reg_t_spin.setRange(0, max(n_t - 1, 0))
        self.reg_t_spin.setToolTip(
            "Which timepoint the pose graph is solved on. Every timepoint is then fused with "
            "that ONE solution, so a drifting stage does not give t=0 and t=9 different "
            "placements.")
        self.reg_t_row = _row(QLabel("Registration timepoint:"), self.reg_t_spin)
        if n_t > 1:
            self.v.addLayout(self.reg_t_row)

        self.rel_spin = QSpinBox()
        self.rel_spin.setRange(1, 200)
        self.rel_spin.setValue(STITCH_DEFAULTS["outlier_rel_pct"])
        self.rel_spin.setSuffix("%")
        self.rel_spin.setToolTip(
            "Blunder rejection, relative term: drop a link whose residual exceeds this "
            "percentage of the median residual.")
        self.abs_spin = QSpinBox()
        self.abs_spin.setRange(1, 50)
        self.abs_spin.setValue(STITCH_DEFAULTS["outlier_abs_px"])
        self.abs_spin.setSuffix(" px")
        self.abs_spin.setToolTip(
            "Blunder rejection, absolute term: a link must ALSO be off by at least this "
            "many pixels to be dropped. Both conditions have to hold, so a very clean plate "
            "does not start rejecting links that were off by a fraction of a pixel.")
        self.v.addLayout(_row(QLabel("Outlier rel:"), self.rel_spin,
                              QLabel("abs:"), self.abs_spin))

        # "Correct lens distortion (per-seam elastic)" -- maragall/stitcher app.py:1472, the
        # control Julio asked about by name.
        #
        # In the reference tool this checkbox is DEAD: `distortion_checkbox` is created and
        # never read, so FusionWorker's enable_distortion default (True) always wins and
        # unchecking it changes nothing. Here it decides, and tests pin both directions.
        self.distortion_cb = QCheckBox("Correct lens distortion (per-seam elastic)")
        self.distortion_cb.setChecked(STITCH_DEFAULTS["correct_distortion"])
        self.distortion_cb.setToolTip(
            "Fit a per-tile elastic warp from the REGISTERED seams and apply it during fusion "
            "(tilefusion.distortion). It corrects what a rigid solve cannot: field curvature "
            "and lens distortion bending each tile, which shows up as seams that are sharp in "
            "the middle and doubled at the ends.\n\n"
            "Needs registration -- it corrects the residual left after the global solve.\n\n"
            "ON by default. Untick it to fuse on the rigid solve alone, which is faster and is "
            "the right control when you want to see what the elastic fit is actually buying.")
        self.v.addWidget(self.distortion_cb)

        # -- fusion --------------------------------------------------------------------
        self.v.addWidget(_head("FUSION"))
        self.blend_spin = QSpinBox()
        self.blend_spin.setRange(1, 2000)
        self.blend_spin.setValue(STITCH_DEFAULTS["blend_px"])
        self.blend_spin.setSuffix(" px")
        self.blend_spin.setToolTip(
            "Hann feather ramp width. It must fit INSIDE the real overlap: on the 10x tissue "
            "set the measured overlap is ~208 px, which is what the 128 px default was sized "
            "against. A ramp wider than the overlap never reaches full weight and dims the "
            "seam.")
        self.blend_auto_cb = QCheckBox("Auto (measure the real overlap)")
        self.blend_auto_cb.setChecked(STITCH_DEFAULTS["auto_blend"])
        self.blend_auto_cb.setToolTip(
            "Measure this acquisition's actual seam overlap and size the ramp to it (median "
            "overlap x 2), instead of the fixed default -- which is sized to ONE acquisition "
            "(~208 px on the 10x tissue set) and is wrong on a denser grid.")
        self.blend_auto_cb.toggled.connect(lambda on: self.blend_spin.setEnabled(not on))
        self.v.addLayout(_row(QLabel("Blend width:"), self.blend_spin, self.blend_auto_cb))

        self.channel_boxes = []
        if names:
            self.v.addWidget(_head("CHANNELS TO FUSE"))
            self.v.addWidget(_wrapped(
                "Every channel is fused with the ONE geometry solved above. This is the "
                "memory lever: a 27-FOV 10x region is ~0.2 GB at one channel and ~0.9 GB at "
                "four.", _SUB))
            box_row = QHBoxLayout()
            box_row.setSpacing(8)
            for name in names:
                cb = QCheckBox(name)
                cb.setChecked(True)
                self.channel_boxes.append(cb)
                box_row.addWidget(cb)
            box_row.addStretch(1)
            self.v.addLayout(box_row)

        # -- run -----------------------------------------------------------------------
        self.run_btn = QPushButton("Run stitcher iteration")
        self.run_btn.setCursor(Qt.PointingHandCursor)
        self.run_btn.setToolTip(
            "Register and fuse the selected scope and show the result. Nothing is written "
            "to disk and the acquisition is never modified.")
        self.run_btn.clicked.connect(self._run)
        self.v.addWidget(self.run_btn)

        self.save_cb = QCheckBox("Also write the fused mosaics to disk (OME-Zarr)")
        self.save_cb.setToolTip(
            "Off by default: tuning a registration/fusion run should cost compute, not disk. "
            "The settings above travel to the saved run too.")
        self.v.addWidget(self.save_cb)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setMaximumHeight(6)
        self.progress.setVisible(False)
        self.v.addWidget(self.progress)
        self.v.addWidget(self.status)
        self.v.addStretch(1)

        _apply_qss(self)
        self._on_register_toggled(self.register_cb.isChecked())
        self._check_projector(self.projector_combo.currentText())

    # -- behaviour ---------------------------------------------------------------------
    def _on_register_toggled(self, on: bool) -> None:
        """Grey out the knobs that provably do nothing with registration off."""
        for w in (self.reg_channel_combo, self.rel_spin, self.abs_spin,
                  self.reg_t_spin, self.distortion_cb):
            w.setEnabled(bool(on))

    def _check_projector(self, name: str) -> None:
        why = stitch_refusal(name)
        self.run_btn.setEnabled(why is None)
        self.say("" if why is None else why)

    def kwargs(self) -> dict:
        """The panel's settings as ``stitch_region`` keyword arguments."""
        meta = getattr(self.host, "_meta", None) or {}
        frame = meta.get("frame_shape") or (None, None)
        selected = [i for i, cb in enumerate(self.channel_boxes) if cb.isChecked()]
        return stitch_operator_kwargs(
            register=self.register_cb.isChecked(),
            registration_channel=(self.reg_channel_combo.currentIndex()
                                  if self.register_cb.isChecked() else None),
            channels=selected if self.channel_boxes else None,
            blend_px=self.blend_spin.value(),
            auto_blend=self.blend_auto_cb.isChecked(),
            correct_distortion=self.distortion_cb.isChecked(),
            registration_t=self.reg_t_spin.value(),
            outlier_rel_pct=self.rel_spin.value(),
            outlier_abs_px=self.abs_spin.value(),
            n_channels=len(self.channel_boxes) or None,
            tile_px=min(frame) if all(frame) else None,
        )

    def _run(self) -> None:
        why = stitch_refusal(self.projector_combo.currentText())
        if why is not None:
            self.say(why)
            return
        try:
            kwargs = self.kwargs()
        except ValueError as exc:                 # a refused setting -> say it, run nothing
            self.say(str(exc))
            return
        kwargs["projector"] = self.projector_combo.currentText()
        self.say("")
        # regions=None means UNSCOPED, not "the whole plate": run_operator resolves it against
        # the run selector and the live selection. See the SCOPE note in __init__.
        self.host.run_operator("stitch", regions=None,
                               save=self.save_cb.isChecked(), operator_kwargs=kwargs)


# ---------------------------------------------------------------------------------------
# 2. deconvolution — the iterative QC loop
# ---------------------------------------------------------------------------------------

class QCFrame(NamedTuple):
    """One finished QC run, as the panel receives it.

    The VOLUME travels with the composite because the picture is clickable: re-sectioning
    through a different point is ``qc_composite`` again on this same array (see
    ``DeconQCResultView._on_image_clicked``), and the alternative — throwing the volume away in
    the worker and re-running RL to look two pixels to the left — is an RL run per click.
    """

    composite: object          # (H, W) float, what qc_composite returned at `centre`
    volume: object             # (Z, Y, X) the deconvolved crop the sections were cut from
    centre: tuple              # (z, y, x) in `volume`: the brightest structure RL was judged at
    view_half: object          # the lateral half-width the composite was cut to, or None


class _DeconQCWorker(QThread):
    """Run RL at ONE iteration count on ONE FOV's z-stack and measure the halo.

    A thread because RL on a 256-px crop is seconds, not milliseconds, and a frozen window
    during a QC loop is the reason nobody runs the QC loop.
    """

    done = Signal(int, object, float)        # (iterations, QCFrame, halo/core ratio)
    failed = Signal(str)

    def __init__(self, dataset, region, fov, channel, iterations, gpu, crop_half, view_half):
        super().__init__()
        self._args = (dataset, region, fov, channel, iterations, gpu, crop_half, view_half)

    def run(self):
        dataset, region, fov, channel, iterations, gpu, crop_half, view_half = self._args
        if self.isInterruptionRequested():
            return          # a close/shutdown beat us to the start — do not begin an RL run nobody
            #                 is waiting for (lets DeconQCPanel.shutdown().wait() return at once)
        try:
            from squidmip._decon import OpticsParams, _run, make_psf
            from squidmip._decon_qc import (
                brightest_structure,
                crop_around,
                halo_core_ratio,
                load_stack,
                qc_composite,
                qc_window_um,
            )

            stack, region, channel, _meta = load_stack(dataset, region, fov, channel)
            optics = OpticsParams.from_acquisition(dataset, channel=channel)
            optics = OpticsParams(optics.na, optics.wavelength_um, optics.dxy_um,
                                  optics.dz_um, int(stack.shape[0]), optics.ni)
            core_um = 0.61 * optics.wavelength_um / optics.na
            window_um = qc_window_um(core_um, stack.shape[0], optics.dz_um)
            z_margin = int(np.ceil(window_um / optics.dz_um))
            centre_full = brightest_structure(stack, optics.dxy_um, optics.dz_um, core_um,
                                              z_margin=z_margin, xy_margin=crop_half)
            crop, centre = crop_around(stack, centre_full, crop_half)
            volume = _run(crop, make_psf(optics), int(iterations), gpu=gpu)
            ratio = halo_core_ratio(volume, centre, optics.dxy_um, optics.dz_um,
                                    core_um, window_um)
            self.done.emit(int(iterations),
                           QCFrame(qc_composite(volume, centre, view_half=view_half),
                                   volume, tuple(int(v) for v in centre), view_half),
                           float(ratio))
        except Exception as exc:                  # reported as a sentence, never swallowed
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class _ClickableImage(QLabel):
    """A QLabel showing an unscaled pixmap, which reports clicks in PIXMAP pixels.

    The label is centred in a scroll area and is usually larger than the picture, so the
    press position has to have the centring offset taken off it before it means anything.
    Clicks outside the pixmap are dropped here rather than sent on as negative coordinates.
    """

    clicked = Signal(int, int)             # (row, col) in the pixmap's own pixels

    def mousePressEvent(self, event):      # noqa: N802 (Qt's spelling)
        pm = self.pixmap()
        if pm is not None and not pm.isNull():
            pos = event.pos()
            dx = max((self.width() - pm.width()) // 2, 0)
            dy = max((self.height() - pm.height()) // 2, 0)
            col, row = pos.x() - dx, pos.y() - dy
            if 0 <= col < pm.width() and 0 <= row < pm.height():
                self.clicked.emit(int(row), int(col))
        super().mousePressEvent(event)


class DeconQCResultView(QWidget):
    """PANE 3. The deconvolved 2-D image in turbo with the x-z and y-z strips attached.

    It RENDERS what :func:`squidmip._decon_qc.qc_composite` and
    :func:`squidmip._decon_qc.turbo_rgb` produced; it builds no picture of its own. A view
    that assembled three panels itself would be a second renderer to keep in step with the
    CLI montage, which is this project's dominant defect shape.

    CLICK TO MOVE THE CROSSHAIRS. Julio: "we should be able to toggle the turbo colormap
    mini-gui where we click on there image and it moves teh crosshairs to display XZ and YZ
    bands." The x-z and y-z strips are sections through ONE point, and the point the QC run
    picked is the brightest structure it found — not necessarily the one worth judging. A
    click re-sections the SAME volume through the clicked point: ``qc_composite`` already
    takes ``centre``, so this is a re-slice of an array already in memory, not another RL
    run. The mapping from pixel to voxel is ``_decon_qc.composite_centre_at``, beside the
    layout it inverts.
    """

    def __init__(self, subject: str):
        super().__init__()
        self.setStyleSheet(f"background:{_BG};color:#e6edf3;")
        self.history: list = []
        # The volume BEHIND the current picture, kept so a click can re-section it. None until
        # a caller passes one (the old three-argument show_iteration still works and simply
        # leaves the picture unclickable — there is nothing to re-slice).
        self._volume = None
        self._centre = None
        self._view_half = None
        self._gap = 2
        v = QVBoxLayout(self)
        v.setContentsMargins(14, 12, 14, 12)
        v.setSpacing(8)
        t = QLabel(f"Deconvolution QC · {subject}")
        t.setStyleSheet("font-size:15px;font-weight:800;")
        v.addWidget(t)
        legend = QLabel("x-y with the y-z strip to its right and the x-z strip below, all "
                        "TURBO on one shared scale. Turbo has a steep ramp through the low "
                        "intensities where a halo lives; on a grey ramp the halo is the part "
                        "of the image the eye is worst at.")
        legend.setWordWrap(True)
        legend.setStyleSheet(_SUB)
        v.addWidget(legend)

        self.caption_label = QLabel("")
        self.caption_label.setWordWrap(True)
        self.caption_label.setStyleSheet("font-size:12px;font-weight:700;")
        v.addWidget(self.caption_label)

        self.verdict_label = QLabel("")
        self.verdict_label.setWordWrap(True)
        self.verdict_label.setStyleSheet("color:#d29922;font-size:11px;")
        v.addWidget(self.verdict_label)

        # WHERE the sections are cut. It is the one thing a moved crosshair changes that the
        # picture alone cannot state, and it is also where the caveat lives: the halo/core number
        # belongs to the structure the run measured, not to wherever the user has clicked since.
        self.crosshair_label = QLabel("")
        self.crosshair_label.setWordWrap(True)
        self.crosshair_label.setStyleSheet(_SUB)
        v.addWidget(self.crosshair_label)

        self.image_label = _ClickableImage()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setCursor(Qt.CrossCursor)
        self.image_label.setToolTip(
            "Click anywhere in the x-y plane, the x-z strip or the y-z strip to move the "
            "crosshairs there. The sections are re-cut from the same deconvolved volume; "
            "nothing is re-run.")
        self.image_label.clicked.connect(self._on_image_clicked)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        scroll.setWidget(self.image_label)
        v.addWidget(scroll, 1)

        self.trail_label = QLabel("")
        self.trail_label.setWordWrap(True)
        self.trail_label.setStyleSheet(_SUB)
        v.addWidget(self.trail_label)

    def show_iteration(self, iterations: int, composite, ratio: float,
                       kind: str, verdict: str, volume=None, centre=None,
                       view_half=None, gap: int = 2) -> None:
        """Display one iteration's composite and remember it, so the loop can be compared.

        *volume*, *centre* and *view_half* are what the composite was cut FROM. Passing them
        makes the picture clickable (see :meth:`_on_image_clicked`); leaving them out shows
        exactly the same picture and simply does not respond to clicks, which is what the
        older three-argument callers get.
        """
        self._volume = None if volume is None else np.asarray(volume)
        self._centre = None if centre is None else tuple(int(v) for v in centre)
        self._view_half = view_half
        self._gap = int(gap)
        self._paint(composite)
        self.history.append((int(iterations), float(ratio)))
        self.caption_label.setText(
            f"{iterations} iteration" + ("s" if iterations != 1 else "")
            + f"  ·  halo/core {ratio:.3f}")
        self.verdict_label.setText(verdict)
        self.trail_label.setText("  ".join(f"k={k}: {r:.3f}" for k, r in self.history))
        self._sync_crosshair_label()

    def _paint(self, composite) -> None:
        """Put one composite on screen. The ONLY place a pixmap is set, so the first paint and
        every crosshair move go through identical code."""
        from squidmip._decon_qc import turbo_rgb

        rgb = np.ascontiguousarray(turbo_rgb(composite))
        h, w, _ = rgb.shape
        image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.image_label.setPixmap(QPixmap.fromImage(image))
        self.image_label.setMinimumSize(w, h)
        self._rgb = rgb                       # keep the buffer alive alongside the pixmap

    def _on_image_clicked(self, row: int, col: int) -> None:
        """Move the crosshairs to the clicked voxel and re-cut the three sections there.

        No RL run, no worker: the deconvolved volume is already in memory and
        :func:`~squidmip._decon_qc.qc_composite` takes the centre, so this is a re-slice.
        A click in a separator band or in the dead corner maps to nothing and is ignored —
        guessing a nearby voxel would move the crosshairs somewhere the user did not point.
        """
        if self._volume is None or self._centre is None:
            return
        from squidmip._decon_qc import composite_centre_at, qc_composite

        centre = composite_centre_at(self._volume.shape, self._centre, row, col,
                                     view_half=self._view_half, gap=self._gap)
        if centre is None or centre == self._centre:
            return
        self._centre = centre
        self._paint(qc_composite(self._volume, centre, view_half=self._view_half,
                                 gap=self._gap))
        self._sync_crosshair_label(moved=True)

    def _sync_crosshair_label(self, moved: bool = False) -> None:
        if self._centre is None:
            self.crosshair_label.setText(
                "" if self._volume is None else "crosshairs: unknown")
            return
        z, y, x = self._centre
        where = f"crosshairs at z={z}, y={y}, x={x}"
        self.crosshair_label.setText(
            where + ("  ·  moved by hand; the halo/core number above was measured at the "
                     "structure the run picked, not here." if moved else
                     "  ·  the brightest structure the run found. Click the picture to "
                     "section somewhere else."))


class DeconQCPanel(_Panel):
    """Pick an iteration count, run it, judge the picture in the tab beside this, add one more."""

    def __init__(self, host):
        super().__init__(
            host, "Deconvolution (Richardson-Lucy)",
            "Richardson-Lucy is SEMI-CONVERGENT: the halo tightens for a few iterations and "
            "then a disc around the core starts growing back as the algorithm fits noise. "
            "There is no universally correct count, so run one, look at the turbo x-z / y-z "
            "view in the tab it opens, then add ONE more and look again.")
        from squidmip._decon import QC_START_ITERATIONS
        from squidmip._decon_qc import DEFAULT_CROP_HALF, DEFAULT_VIEW_HALF

        self._crop_half = DEFAULT_CROP_HALF
        self._view_half = DEFAULT_VIEW_HALF
        self._worker = None
        self._view = None
        self._view_subject = None

        self.v.addWidget(_head("WHERE TO MEASURE"))
        self.region_combo = QComboBox()
        for region in getattr(host, "_order", []):
            self.region_combo.addItem(region)
        self.fov_spin = QSpinBox()
        self.fov_spin.setRange(0, 9999)
        self.fov_spin.setToolTip(
            "The QC runs on ONE FOV. A recommendation is for THIS sample at THIS exposure — "
            "SNR and structure decide the answer, so it is never a global default.")
        self.v.addLayout(_row(QLabel("Region:"), self.region_combo,
                              QLabel("FOV:"), self.fov_spin))

        self.channel_combo = QComboBox()
        for name in _channel_names(host):
            self.channel_combo.addItem(name)
        self.channel_combo.setToolTip(
            "The emission wavelength of this channel sets the PSF. The kernel is a VECTORIAL "
            "PSF computed from the acquisition's own optics, not a Gaussian.")
        self.v.addLayout(_row(QLabel("Channel:"), self.channel_combo))

        self.v.addWidget(_head("ITERATIONS"))
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(1, 100)
        self.iter_spin.setValue(QC_START_ITERATIONS)
        self.iter_spin.setToolTip("Richardson-Lucy iterations for the next run.")
        self.iter_hint = QLabel(f"shipped default: {_shipped_iterations()}")
        self.iter_hint.setStyleSheet(_SUB)
        self.plus_btn = QPushButton("+1 iteration")
        self.plus_btn.setToolTip(
            "Add exactly one and re-run. One at a time is the point: the turn is judged by "
            "eye between steps, and a jump of five hides where it happened.")
        self.plus_btn.clicked.connect(
            lambda: self.iter_spin.setValue(self.iter_spin.value() + 1))
        self.v.addLayout(_row(QLabel("Run with:"), self.iter_spin, self.iter_hint,
                              self.plus_btn))

        self.cpu_cb = QCheckBox("Force CPU (disable GPU)")
        self.cpu_cb.setToolTip(
            "Selects a BACKEND, not an algorithm — the RL update is identical either way.")
        self.v.addWidget(self.cpu_cb)

        self.run_btn = QPushButton("Deconvolve and show the turbo x-z / y-z view")
        self.run_btn.setCursor(Qt.PointingHandCursor)
        self.run_btn.clicked.connect(self.run)
        self.v.addWidget(self.run_btn)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setMaximumHeight(6)
        self.progress.setVisible(False)
        self.v.addWidget(self.progress)
        self.v.addWidget(self.status)

        note = _wrapped("The result opens as a tab beside this one. Nothing is written next to "
                        "the acquisition; the datasets are opened read only.", _SUB)
        self.v.addWidget(note)
        self.v.addStretch(1)
        _apply_qss(self)

    # -- behaviour ---------------------------------------------------------------------
    def _subject(self) -> str:
        return (f"{self.region_combo.currentText()}/{self.fov_spin.value()}/"
                f"{self.channel_combo.currentText()}")

    def run(self) -> None:
        dataset = getattr(self.host, "_acq_path", None)
        if not dataset:
            self.say("no acquisition is open — deconvolution needs the dataset folder to read "
                     "its optics (NA, emission wavelength, pixel size, z-step) from.")
            return
        if self._worker is not None and self._worker.isRunning():
            self.say("a deconvolution is already running — let it finish before adding an "
                     "iteration.")
            return
        if not hasattr(self.host, "publish_qc_result"):
            self.say("this window cannot show a QC result: it does not implement "
                     "publish_qc_result(widget, title), which is how it is handed a result "
                     "tab. Refusing to deconvolve and then drop the picture.")
            return

        iterations = self.iter_spin.value()
        subject = self._subject()
        if self._view is None or self._view_subject != subject:
            self._view = DeconQCResultView(subject)
            self._view_subject = subject
            self.host.publish_qc_result(self._view, f"Decon QC · {subject}")

        self.progress.setVisible(True)
        self.run_btn.setEnabled(False)
        self.say(f"deconvolving {subject} at {iterations} iterations …")
        self._worker = _DeconQCWorker(
            dataset, self.region_combo.currentText(), self.fov_spin.value(),
            self.channel_combo.currentText(), iterations, not self.cpu_cb.isChecked(),
            self._crop_half, self._view_half)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_done(self, iterations, frame, ratio) -> None:
        from squidmip._decon_qc import halo_verdict

        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)
        # The verdict needs THIS iteration included, and show_iteration is what appends it,
        # so build the history to judge on here rather than reading it back afterwards.
        history = list(self._view.history) + [(int(iterations), float(ratio))]
        kind, verdict = halo_verdict(history)
        self._view.show_iteration(iterations, frame.composite, ratio, kind, verdict,
                                  volume=frame.volume, centre=frame.centre,
                                  view_half=frame.view_half)
        self.say(verdict)

    def _on_failed(self, message: str) -> None:
        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)
        self.say(f"deconvolution did not run: {message}")

    def shutdown(self) -> None:
        """Join the QC worker before this panel is destroyed.

        ``_dispose_tab_widget`` (tab close / float close / app exit) calls ``shutdown()`` on any
        panel that has one, then ``deleteLater()``s it. This panel used to expose only ``stop()``
        — which the teardown path does not call — and even that waited a mere 50 ms, far less than
        an RL run. So closing the Decon QC tab mid-run dropped the last reference to a RUNNING
        QThread: "QThread: Destroyed while thread is still running" aborts the interpreter. RL on a
        256 px crop is a bounded, fixed-iteration run, so waiting for it to finish is finite.
        """
        w = self._worker
        if w is None:
            return
        # Drop the result callbacks FIRST: a done/failed emit that lands while this panel and its
        # pane-3 view are being torn down would call show_iteration on a deleted widget.
        for sig, slot in ((w.done, self._on_done), (w.failed, self._on_failed)):
            try:
                sig.disconnect(slot)
            except (TypeError, RuntimeError):
                pass          # not connected, or already gone — either way there is nothing to drop
        if w.isRunning():
            w.requestInterruption()   # honoured before the expensive RL call begins
            w.wait()                  # block until run() returns — bounded on a crop, never a hang
        self._worker = None

    # ``stop`` predates ``shutdown`` and joined for only 50 ms (far less than an RL run), so the
    # thread it meant to reap was usually still alive. It now routes through the real join.
    def stop(self) -> None:
        self.shutdown()


def _shipped_iterations() -> int:
    from squidmip._decon import DEFAULT_ITERATIONS

    return DEFAULT_ITERATIONS
